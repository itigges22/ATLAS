package main

import (
	"strings"
	"testing"
)

func TestIsLoopingTail(t *testing.T) {
	loop := "The first line is <!DOCTYPE html>. " +
		strings.Repeat("Wait, I'll check if I can see the output. I can't. I'll just say it. ", 6)
	if !isLoopingTail(loop) {
		t.Errorf("expected a repeating self-doubt stream to be detected as a loop")
	}
	normal := "The first line of index.html is `<!DOCTYPE html>`. It declares the document type for the HTML5 page, followed by the html and head elements with meta tags and a title."
	if isLoopingTail(normal) {
		t.Errorf("a normal varied response must not be flagged as a loop")
	}
}
