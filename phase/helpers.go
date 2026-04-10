package phase

import (
	"strconv"
)

// itoa is a convenience alias.
func itoa(n int) string {
	return strconv.Itoa(n)
}

// intSliceToStrings converts []int to []string for ApplyAxisOpts.
func intSliceToStrings(ints []int) []string {
	s := make([]string, len(ints))
	for i, v := range ints {
		s[i] = strconv.Itoa(v)
	}
	return s
}

// stringsToIntSlice converts []string to []int (skips non-numeric).
func stringsToIntSlice(ss []string) []int {
	var result []int
	for _, s := range ss {
		if n, err := strconv.Atoi(s); err == nil {
			result = append(result, n)
		}
	}
	return result
}

// dedupeInts removes duplicates from a sorted-or-unsorted int slice,
// preserving order of first occurrence.
func dedupeInts(vals []int) []int {
	seen := make(map[int]bool)
	var result []int
	for _, v := range vals {
		if !seen[v] {
			seen[v] = true
			result = append(result, v)
		}
	}
	return result
}

// containsInt returns true if v is in slice.
func containsInt(slice []int, v int) bool {
	for _, s := range slice {
		if s == v {
			return true
		}
	}
	return false
}

