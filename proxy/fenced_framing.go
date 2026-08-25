package main

import "strings"

// Fenced-payload framing.
//
// "@fenced" routes a file body around the JSON channel, and the parent then
// has to answer one question about whatever came back: is this a whole file?
//
// It used to answer that from a single signal -- an opening fence with no
// closing fence -- and treat everything else as complete. "Everything else"
// includes a bare body with no fence at all, which proves nothing in either
// direction. The sealed Stage-A acquisition inlined 42 bodies and NOT ONE of
// them opened a fence, so the guard could not fire on any of them; one had
// stopped mid-emission and 1106 unframed bytes were written, parsed cleanly,
// and did nothing. A syntax gate cannot catch that, because a body cut in the
// middle of a comment is valid in most languages.
//
// So framing is the only evidence used here. A payload is a file when its
// protocol framing says the emission finished: an opening fence and a closing
// fence. Nothing below inspects the body -- not its language, syntax, final
// line, comments, indentation, prose, or length. A completed emission of
// something that looks unfinished is a file; an unterminated emission of
// something that looks finished is not.
type fenceFraming int

const (
	// fenceFramingComplete: opening and closing fences both present. The
	// body between them is the file.
	fenceFramingComplete fenceFraming = iota
	// fenceFramingUnterminated: fence markers are present but do not close a
	// block. The emission was cut, or never framed what followed.
	fenceFramingUnterminated
	// fenceFramingAbsent: no fence markers at all. There is no framing
	// evidence, so completion cannot be established either way.
	fenceFramingAbsent
)

func (f fenceFraming) String() string {
	switch f {
	case fenceFramingComplete:
		return "complete"
	case fenceFramingUnterminated:
		return "fence opened, never closed"
	default:
		return "no fence at all"
	}
}

// classifyFencedPayload is the ONE framing decision. The inline path and the
// sub-call path both route through it, so the two cannot drift into
// disagreeing about what counts as a finished emission.
//
// A body is returned only for fenceFramingComplete. Every other outcome
// returns "" so that no caller can accidentally use bytes whose framing did
// not prove they are whole.
func classifyFencedPayload(payload string) (fenceFraming, string) {
	if body := extractFencedContent(payload); body != "" {
		return fenceFramingComplete, body
	}
	if strings.Contains(payload, "```") {
		return fenceFramingUnterminated, ""
	}
	return fenceFramingAbsent, ""
}

// resolveInlineFencedBody decides whether the bytes the model inlined after
// the "@fenced" sentinel may be used as the file.
//
// ok is true only when framing proved the emission finished. Otherwise the
// caller falls back to the sub-call -- the channel built to carry a file
// body -- and why names the framing truthfully for the log.
func resolveInlineFencedBody(inline string) (body string, ok bool, why fenceFraming) {
	framing, resolved := classifyFencedPayload(inline)
	if framing == fenceFramingComplete {
		return resolved, true, framing
	}
	return "", false, framing
}
