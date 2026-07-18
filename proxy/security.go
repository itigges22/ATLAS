package main

import "strconv"

// safeLogField encodes untrusted text as one quoted ASCII log field. Newlines,
// carriage returns, and control bytes become escape sequences, so model/user
// data cannot forge additional log records.
func safeLogField(value string, maxLen int) string {
	if maxLen > 0 {
		value = truncateStr(value, maxLen)
	}
	return strconv.QuoteToASCII(value)
}
