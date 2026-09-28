package fusion

import (
	"sort"
	"time"
)

// ReorderBuffer 乱序观测重排缓冲：按 (有效时间键, 序号) 全序出队，
// 重复或倒退被确定且可区分地处理，绝不静默丢弃。
//
// 有效时间键由 keyOf 提供（在线对齐后的统一时间）；未提供时退化为
// 观测原始时间戳。键在比较时按当前时钟模型实时计算——由于模型始终
// 是严格单调仿射映射（skew>0），任意两条观测的键序与源时间戳序一致，
// 因而在乱序到达、模型在线更新的情况下排序仍然确定且可复现。
type ReorderBuffer struct {
	maxSize    int
	items      []Observation
	lastOut    Observation // 最近一次出队的观测
	lastOutKey time.Time   // 最近出队观测在出队时刻的有效键
	hasOut     bool
	keyOf      func(Observation) time.Time
}

// NewReorderBuffer 创建容量受限的重排缓冲。可选传入有效键函数。
func NewReorderBuffer(maxSize int, keyOf ...func(Observation) time.Time) *ReorderBuffer {
	if maxSize < 1 {
		maxSize = 1
	}
	b := &ReorderBuffer{maxSize: maxSize}
	if len(keyOf) > 0 {
		b.keyOf = keyOf[0]
	}
	return b
}

// key 返回观测当前的有效时间键（对齐后统一时间）。
func (b *ReorderBuffer) key(o Observation) time.Time {
	if b.keyOf != nil {
		return b.keyOf(o)
	}
	return o.Stamp
}

// less 定义全序：先有效时间键，后序号，再传感器类型，保证确定性。
func (b *ReorderBuffer) less(a, bk Observation) bool {
	ka, kb := b.key(a), b.key(bk)
	if !ka.Equal(kb) {
		return ka.Before(kb)
	}
	if a.Seq != bk.Seq {
		return a.Seq < bk.Seq
	}
	return a.Kind < bk.Kind
}

// Push 插入观测。返回是否接收；缓冲满返回 false（由上层按策略拒绝）。
func (b *ReorderBuffer) Push(o Observation) bool {
	if len(b.items) >= b.maxSize {
		return false
	}
	b.items = append(b.items, o)
	return true
}

// Pop 弹出全序最小的观测。第二个返回值标识是否存在。
func (b *ReorderBuffer) Pop() (Observation, bool) {
	if len(b.items) == 0 {
		return Observation{}, false
	}
	i := 0
	for j := 1; j < len(b.items); j++ {
		if b.less(b.items[j], b.items[i]) {
			i = j
		}
	}
	o := b.items[i]
	b.items = append(b.items[:i], b.items[i+1:]...)
	b.lastOut = o
	b.lastOutKey = b.key(o)
	b.hasOut = true
	return o, true
}

// Peek 查看全序最小的观测及其当前有效键而不弹出。
func (b *ReorderBuffer) Peek() (Observation, bool) {
	if len(b.items) == 0 {
		return Observation{}, false
	}
	i := 0
	for j := 1; j < len(b.items); j++ {
		if b.less(b.items[j], b.items[i]) {
			i = j
		}
	}
	return b.items[i], true
}

// PeekKey 查看队首观测的当前有效键；缓冲空返回零值与 false。
func (b *ReorderBuffer) PeekKey() (time.Time, bool) {
	o, ok := b.Peek()
	if !ok {
		return time.Time{}, false
	}
	return b.key(o), true
}

// Classify 判定观测相对已出队序列的关系（以出队时冻结的键为准，
// 避免模型在线更新后把新观测误判为倒退）。
// 键早于最近出队：倒退；键相同且序号不更大：重复。
func (b *ReorderBuffer) Classify(o Observation) RejectReason {
	if !b.hasOut {
		return ReasonAccepted
	}
	k := b.key(o)
	if k.Before(b.lastOutKey) {
		return ReasonTimestampRegress
	}
	if k.Equal(b.lastOutKey) {
		if o.Seq <= b.lastOut.Seq && o.Kind == b.lastOut.Kind {
			return ReasonDuplicate
		}
	}
	return ReasonAccepted
}

// Len 当前缓冲长度。
func (b *ReorderBuffer) Len() int { return len(b.items) }

// FlushBefore 弹出所有有效键早于 t 的观测（供降级清理），按全序返回。
func (b *ReorderBuffer) FlushBefore(t time.Time) []Observation {
	var out []Observation
	var keep []Observation
	for _, o := range b.items {
		if b.key(o).Before(t) {
			out = append(out, o)
		} else {
			keep = append(keep, o)
		}
	}
	b.items = keep
	sort.Slice(out, func(i, j int) bool { return b.less(out[i], out[j]) })
	return out
}

// MaxStamp 返回缓冲内最大有效时间键；空缓冲返回零值。
func (b *ReorderBuffer) MaxStamp() time.Time {
	var max time.Time
	for _, o := range b.items {
		if k := b.key(o); k.After(max) {
			max = k
		}
	}
	return max
}

// Contains 判定缓冲内是否已存在相同 (类型, 原始时间戳, 序号) 的观测。
// 以原始身份判定，不受模型更新影响。
func (b *ReorderBuffer) Contains(o Observation) bool {
	for _, it := range b.items {
		if it.Kind == o.Kind && it.Stamp.Equal(o.Stamp) && it.Seq == o.Seq {
			return true
		}
	}
	return false
}
