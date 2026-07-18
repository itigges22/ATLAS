package main

import (
	"strings"
	"testing"
)

// closestLineHint maps a from-memory paraphrase onto the real file line
// so the edit_file mismatch error can anchor the model's retry in actual
// content. The motivating case (Jun 8 2026 flask test): the model
// searched for `item = items[id + 1]` when the file's real line was
// `return jsonify(items[item_id + 1])`, then gave up on the surgical
// edit and rewrote the whole function from the same faulty memory.
func TestClosestLineHintParaphrase(t *testing.T) {
	content := `from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/items/<int:item_id>", methods=["GET"])
def get_item(item_id):
    return jsonify(items[item_id + 1])
`
	hint := closestLineHint(content, "item = items[id + 1]")
	if hint == "" {
		t.Fatal("expected a hint for a near-miss paraphrase, got none")
	}
	if !strings.Contains(hint, "jsonify(items[item_id + 1])") {
		t.Errorf("hint should quote the real line, got: %s", hint)
	}
	if !strings.Contains(hint, "line 7") {
		t.Errorf("hint should carry the real line number, got: %s", hint)
	}
}

func TestClosestLineHintUnrelatedReturnsNothing(t *testing.T) {
	content := "def alpha():\n    return 1\n"
	if hint := closestLineHint(content, "zebra elephant giraffe"); hint != "" {
		t.Errorf("unrelated old_str must not get an anchor, got: %s", hint)
	}
}

func TestClosestLineHintEmptyOldStr(t *testing.T) {
	if hint := closestLineHint("x = 1\n", "   \n  "); hint != "" {
		t.Errorf("blank old_str must not get an anchor, got: %s", hint)
	}
}
