package fusion

import (
	"sort"
	"time"
)

// ReorderBuffer 乱序观测重排缓冲：按 (对齐后统一时间, 序号, 路别)
// 全序出队。所有顺序判定以对齐后的统一时间为准；重复/倒退的身份
// 判定用各路原始时间戳（同一物理时钟内才有可比性）。
type ReorderBuffer struct {
	maxSize int
	items   []pendingItem
	lastOut pendingItem // 最近一次出队的观测
	hasOut  bool
}

// NewReorderBuffer 创建容量受限的重排缓冲。
func NewReorderBuffer(maxSize int) *ReorderBuffer {
	if maxSize < 1 {
		maxSize = 1
	}
	return &ReorderBuffer{maxSize: maxSize}
}

// less 定义全序：先统一时间，后序号，再传感器类型，保证确定性。
func less(a, b pendingItem) bool {
	if !a.aligned.Equal(b.aligned) {
		return a.aligned.Before(b.aligned)
	}
	if a.obs.Seq != b.obs.Seq {
		return a.obs.Seq < b.obs.Seq
	}
	return a.kind < b.kind
}

// Push 插入待定观测。返回是否接收；缓冲满返回 false（由上层按策略拒绝）。
func (b *ReorderBuffer) Push(p pendingItem) bool {
	if len(b.items) >= b.maxSize {
		return false
	}
	b.items = append(b.items, p)
	return true
}

// Pop 弹出全序最小的观测。第二个返回值标识是否存在。
func (b *ReorderBuffer) Pop() (pendingItem, bool) {
	if len(b.items) == 0 {
		return pendingItem{}, false
	}
	i := b.minIndex()
	o := b.items[i]
	b.items = append(b.items[:i], b.items[i+1:]...)
	b.lastOut = o
	b.hasOut = true
	return o, true
}

// Peek 查看全序最小的观测而不弹出。
func (b *ReorderBuffer) Peek() (pendingItem, bool) {
	if len(b.items) == 0 {
		return pendingItem{}, false
	}
	return b.items[b.minIndex()], true
}

func (b *ReorderBuffer) minIndex() int {
	i := 0
	for j := 1; j < len(b.items); j++ {
		if less(b.items[j], b.items[i]) {
			i = j
		}
	}
	return i
}

// Len 当前缓冲长度。
func (b *ReorderBuffer) Len() int { return len(b.items) }

// Index 按下标取条目（不弹出）。
func (b *ReorderBuffer) Index(i int) pendingItem { return b.items[i] }

// SetIndex 覆盖指定下标条目（模型重算后更新）。
func (b *ReorderBuffer) SetIndex(i int, p pendingItem) { b.items[i] = p }

// Remove 删除指定身份的条目（批回滚用）。
func (b *ReorderBuffer) Remove(kind SensorKind, raw time.Time, seq uint64) bool {
	for i, p := range b.items {
		if p.kind == kind && p.raw.Equal(raw) && p.obs.Seq == seq {
			b.items = append(b.items[:i], b.items[i+1:]...)
			return true
		}
	}
	return false
}

// MinReadyIndex 返回就绪且统一时间不晚于 limit 的全序最小条目下标；
// limit 为零值时不施加上界。未就绪（对齐尚不可信）的条目被跳过，
// 但因其在全序上通常更早，会天然挡住后面就绪条目的出队。
func (b *ReorderBuffer) MinReadyIndex(limit time.Time) (int, bool) {
	best := -1
	for i, p := range b.items {
		if !p.ready {
			continue
		}
		if !limit.IsZero() && p.aligned.After(limit) {
			continue
		}
		if best == -1 || less(p, b.items[best]) {
			best = i
		}
	}
	return best, best >= 0
}

// PopIndex 弹出指定下标条目并记录出队序列。
func (b *ReorderBuffer) PopIndex(i int) (pendingItem, bool) {
	if i < 0 || i >= len(b.items) {
		return pendingItem{}, false
	}
	o := b.items[i]
	b.items = append(b.items[:i], b.items[i+1:]...)
	b.lastOut = o
	b.hasOut = true
	return o, true
}

// MaxReadyAligned 返回就绪条目中的最大统一时间。
func (b *ReorderBuffer) MaxReadyAligned() (time.Time, bool) {
	var max time.Time
	found := false
	for _, p := range b.items {
		if p.ready && (!found || p.aligned.After(max)) {
			max, found = p.aligned, true
		}
	}
	return max, found
}

// MaxAligned 返回缓冲内最大统一时间；空缓冲返回零值。
func (b *ReorderBuffer) MaxAligned() time.Time {
	var max time.Time
	for _, p := range b.items {
		if p.aligned.After(max) {
			max = p.aligned
		}
	}
	return max
}

// ContainsRaw 判定缓冲内是否已存在相同 (路别, 原始时间戳, 序号) 的观测。
func (b *ReorderBuffer) ContainsRaw(kind SensorKind, raw time.Time, seq uint64) bool {
	for _, p := range b.items {
		if p.kind == kind && p.raw.Equal(raw) && p.obs.Seq == seq {
			return true
		}
	}
	return false
}

// ClassifyRaw 以各路原始时间戳判定观测相对已出队序列的关系。
// 仅与同一路的最近出队观测比较：原始时间更早为倒退；原始时间相同
// 且序号不更大为重复。
func (b *ReorderBuffer) ClassifyRaw(kind SensorKind, raw time.Time, seq uint64) RejectReason {
	if !b.hasOut || b.lastOut.kind != kind {
		return ReasonAccepted
	}
	if raw.Before(b.lastOut.raw) {
		return ReasonTimestampRegress
	}
	if raw.Equal(b.lastOut.raw) && seq <= b.lastOut.obs.Seq {
		return ReasonDuplicate
	}
	return ReasonAccepted
}

// RemapAll 在某路时钟映射变化后，重算该路全部待定观测的对齐时间
// 与就绪状态，返回重算后的条目（调用方用 fn 的结果替换）。
func (b *ReorderBuffer) RemapAll(a *Alignment) {
	for i := range b.items {
		b.items[i] = a.Remap(b.items[i])
	}
}

// NotReadyCount 返回尚未对齐可信、不能参与消费的条目数。
func (b *ReorderBuffer) NotReadyCount() int {
	n := 0
	for _, p := range b.items {
		if !p.ready {
			n++
		}
	}
	return n
}

// FlushBefore 弹出所有统一时间早于 t 的观测（供降级/清理），按全序返回。
func (b *ReorderBuffer) FlushBefore(t time.Time) []pendingItem {
	var out []pendingItem
	var keep []pendingItem
	for _, p := range b.items {
		if p.aligned.Before(t) {
			out = append(out, p)
		} else {
			keep = append(keep, p)
		}
	}
	b.items = keep
	sort.Slice(out, func(i, j int) bool { return less(out[i], out[j]) })
	return out
}
