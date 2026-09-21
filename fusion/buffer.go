package fusion

import (
	"sort"
	"time"
)

// ReorderBuffer 乱序观测重排缓冲：按 (时间戳, 序号) 全序出队，
// 重复或倒退时间戳被确定且可区分地处理，绝不静默丢弃。
type ReorderBuffer struct {
	maxSize int
	items   []Observation
	lastOut Observation // 最近一次出队的观测
	hasOut  bool
}

// NewReorderBuffer 创建容量受限的重排缓冲。
func NewReorderBuffer(maxSize int) *ReorderBuffer {
	if maxSize < 1 {
		maxSize = 1
	}
	return &ReorderBuffer{maxSize: maxSize}
}

// less 定义全序：先时间戳，后序号，再传感器类型，保证确定性。
func less(a, b Observation) bool {
	if !a.Stamp.Equal(b.Stamp) {
		return a.Stamp.Before(b.Stamp)
	}
	if a.Seq != b.Seq {
		return a.Seq < b.Seq
	}
	return a.Kind < b.Kind
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
		if less(b.items[j], b.items[i]) {
			i = j
		}
	}
	o := b.items[i]
	b.items = append(b.items[:i], b.items[i+1:]...)
	b.lastOut = o
	b.hasOut = true
	return o, true
}

// Classify 判定观测相对已出队序列的关系。
// 时间戳早于最近出队：倒退；时间戳相同且序号不更大：重复。
func (b *ReorderBuffer) Classify(o Observation) RejectReason {
	if !b.hasOut {
		return ReasonAccepted
	}
	if o.Stamp.Before(b.lastOut.Stamp) {
		return ReasonTimestampRegress
	}
	if o.Stamp.Equal(b.lastOut.Stamp) {
		if o.Seq <= b.lastOut.Seq && o.Kind == b.lastOut.Kind {
			return ReasonDuplicate
		}
	}
	return ReasonAccepted
}

// Len 当前缓冲长度。
func (b *ReorderBuffer) Len() int { return len(b.items) }

// FlushBefore 弹出所有时间戳早于 t 的观测（供降级清理），按全序返回。
func (b *ReorderBuffer) FlushBefore(t time.Time) []Observation {
	var out []Observation
	var keep []Observation
	for _, o := range b.items {
		if o.Stamp.Before(t) {
			out = append(out, o)
		} else {
			keep = append(keep, o)
		}
	}
	b.items = keep
	sort.Slice(out, func(i, j int) bool { return less(out[i], out[j]) })
	return out
}

// Peek 查看全序最小的观测而不弹出。
func (b *ReorderBuffer) Peek() (Observation, bool) {
	if len(b.items) == 0 {
		return Observation{}, false
	}
	i := 0
	for j := 1; j < len(b.items); j++ {
		if less(b.items[j], b.items[i]) {
			i = j
		}
	}
	return b.items[i], true
}

// MaxStamp 返回缓冲内最大时间戳；空缓冲返回零值。
func (b *ReorderBuffer) MaxStamp() time.Time {
	var max time.Time
	for _, o := range b.items {
		if o.Stamp.After(max) {
			max = o.Stamp
		}
	}
	return max
}

// Contains 判定缓冲内是否已存在相同 (类型, 时间戳, 序号) 的观测。
func (b *ReorderBuffer) Contains(o Observation) bool {
	for _, it := range b.items {
		if it.Kind == o.Kind && it.Stamp.Equal(o.Stamp) && it.Seq == o.Seq {
			return true
		}
	}
	return false
}
