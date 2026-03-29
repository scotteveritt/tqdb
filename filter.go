package tqdb

import "strings"

// Filter is a predicate over document data fields.
// Matches VS2's MongoDB-style filter operators:
// $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or, $contains
type Filter interface {
	Match(data map[string]any) bool
}

// --- Comparison operators ---

type eqFilter struct {
	field string
	value any
}

// Eq returns a filter that matches when field == value.
// Supports string, float64, bool, and int types.
func Eq(field string, value any) Filter {
	return &eqFilter{field: field, value: value}
}

func (f *eqFilter) Match(data map[string]any) bool {
	if data == nil {
		return false
	}
	v, ok := data[f.field]
	if !ok {
		return false
	}
	return anyEquals(v, f.value)
}

type neFilter struct {
	field string
	value any
}

// Ne returns a filter that matches when field != value.
func Ne(field string, value any) Filter {
	return &neFilter{field: field, value: value}
}

func (f *neFilter) Match(data map[string]any) bool {
	if data == nil {
		return false
	}
	v, ok := data[f.field]
	if !ok {
		return false
	}
	return !anyEquals(v, f.value)
}

// --- Numeric comparison operators ---

type gtFilter struct {
	field string
	value float64
}

// Gt returns a filter that matches when field > value (numeric).
func Gt(field string, value float64) Filter {
	return &gtFilter{field: field, value: value}
}

func (f *gtFilter) Match(data map[string]any) bool {
	if data == nil {
		return false
	}
	fv, ok := toFloat64(data[f.field])
	if !ok {
		return false
	}
	return fv > f.value
}

type gteFilter struct {
	field string
	value float64
}

// Gte returns a filter that matches when field >= value (numeric).
func Gte(field string, value float64) Filter {
	return &gteFilter{field: field, value: value}
}

func (f *gteFilter) Match(data map[string]any) bool {
	if data == nil {
		return false
	}
	fv, ok := toFloat64(data[f.field])
	if !ok {
		return false
	}
	return fv >= f.value
}

type ltFilter struct {
	field string
	value float64
}

// Lt returns a filter that matches when field < value (numeric).
func Lt(field string, value float64) Filter {
	return &ltFilter{field: field, value: value}
}

func (f *ltFilter) Match(data map[string]any) bool {
	if data == nil {
		return false
	}
	fv, ok := toFloat64(data[f.field])
	if !ok {
		return false
	}
	return fv < f.value
}

type lteFilter struct {
	field string
	value float64
}

// Lte returns a filter that matches when field <= value (numeric).
func Lte(field string, value float64) Filter {
	return &lteFilter{field: field, value: value}
}

func (f *lteFilter) Match(data map[string]any) bool {
	if data == nil {
		return false
	}
	fv, ok := toFloat64(data[f.field])
	if !ok {
		return false
	}
	return fv <= f.value
}

// --- Set membership operators ---

type inFilter struct {
	field  string
	values []any
}

// In returns a filter that matches when the field value is one of the given values.
func In(field string, values ...any) Filter {
	return &inFilter{field: field, values: values}
}

func (f *inFilter) Match(data map[string]any) bool {
	if data == nil {
		return false
	}
	v, ok := data[f.field]
	if !ok {
		return false
	}
	for _, candidate := range f.values {
		if anyEquals(v, candidate) {
			return true
		}
	}
	return false
}

type ninFilter struct {
	field  string
	values []any
}

// Nin returns a filter that matches when the field value is NOT one of the given values.
func Nin(field string, values ...any) Filter {
	return &ninFilter{field: field, values: values}
}

func (f *ninFilter) Match(data map[string]any) bool {
	if data == nil {
		return false
	}
	v, ok := data[f.field]
	if !ok {
		return false
	}
	for _, candidate := range f.values {
		if anyEquals(v, candidate) {
			return false
		}
	}
	return true
}

// --- String operators ---

type containsFilter struct {
	field  string
	substr string
}

// Contains returns a filter that matches when the string field value contains the substring.
func Contains(field, substr string) Filter {
	return &containsFilter{field: field, substr: substr}
}

func (f *containsFilter) Match(data map[string]any) bool {
	if data == nil {
		return false
	}
	v, ok := data[f.field]
	if !ok {
		return false
	}
	s, ok := v.(string)
	if !ok {
		return false
	}
	return strings.Contains(s, f.substr)
}

type notContainsFilter struct {
	field  string
	substr string
}

// NotContains returns a filter that matches when the string field value does NOT contain the substring.
func NotContains(field, substr string) Filter {
	return &notContainsFilter{field: field, substr: substr}
}

func (f *notContainsFilter) Match(data map[string]any) bool {
	if data == nil {
		return false
	}
	v, ok := data[f.field]
	if !ok {
		return false
	}
	s, ok := v.(string)
	if !ok {
		return true // non-string field doesn't contain the substring
	}
	return !strings.Contains(s, f.substr)
}

// --- Logical operators ---

type andFilter struct {
	filters []Filter
}

// And returns a filter that matches when ALL sub-filters match (short-circuits on first false).
func And(filters ...Filter) Filter {
	return &andFilter{filters: filters}
}

func (f *andFilter) Match(data map[string]any) bool {
	for _, sub := range f.filters {
		if !sub.Match(data) {
			return false
		}
	}
	return true
}

type orFilter struct {
	filters []Filter
}

// Or returns a filter that matches when ANY sub-filter matches (short-circuits on first true).
func Or(filters ...Filter) Filter {
	return &orFilter{filters: filters}
}

func (f *orFilter) Match(data map[string]any) bool {
	for _, sub := range f.filters {
		if sub.Match(data) {
			return true
		}
	}
	return false
}

// --- helpers ---

// anyEquals compares two any values without fmt.Sprint.
// Handles the common types: string, float64, int, bool.
// Zero allocations on the hot path.
func anyEquals(a, b any) bool {
	switch av := a.(type) {
	case string:
		bv, ok := b.(string)
		return ok && av == bv
	case float64:
		bv, ok := toFloat64(b)
		return ok && av == bv
	case bool:
		bv, ok := b.(bool)
		return ok && av == bv
	case int:
		bv, ok := toFloat64(b)
		return ok && float64(av) == bv
	default:
		// Fallback: compare via toFloat64 for numeric types
		af, aOk := toFloat64(a)
		bf, bOk := toFloat64(b)
		if aOk && bOk {
			return af == bf
		}
		return false
	}
}

// toFloat64 attempts to coerce a value to float64 for numeric comparisons.
func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int8:
		return float64(n), true
	case int16:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	case uint:
		return float64(n), true
	case uint8:
		return float64(n), true
	case uint16:
		return float64(n), true
	case uint32:
		return float64(n), true
	case uint64:
		return float64(n), true
	default:
		return 0, false
	}
}
