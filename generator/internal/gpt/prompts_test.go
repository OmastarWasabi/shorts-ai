package gpt

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadPrompts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompts.txt")
	content := "\uFEFF\r\n[01]\r\nFirst line.\r\n\r\nSecond paragraph.\r\n[02]\r\nAnother prompt.\r\n"
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
	prompts, err := ReadPrompts(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(prompts) != 2 || prompts[0].Number != 1 || prompts[0].Text != "First line.\n\nSecond paragraph." || prompts[1].Number != 2 || prompts[1].Text != "Another prompt." {
		t.Fatalf("unexpected prompts: %+v", prompts)
	}
}

func TestReadPromptsRejectsInvalidInput(t *testing.T) {
	cases := map[string]string{
		"empty":           " \n",
		"missing marker":  "A prompt.",
		"wrong first":     "[02]\nA prompt.",
		"duplicate":       "[01]\nFirst.\n[01]\nSecond.",
		"gap":             "[01]\nFirst.\n[03]\nThird.",
		"empty middle":    "[01]\n\n[02]\nSecond.",
		"empty last":      "[01]\nFirst.\n[02]\n",
		"short marker":    "[1]\nFirst.",
		"inline marker":   "[01]\nFirst.\n[02] Second.",
		"broken marker":   "[01]\nFirst.\n[02\nSecond.",
		"invalid UTF-8":   "[01]\n\xff",
		"oversized text":  "[01]\n" + strings.Repeat("я", 32001),
		"overflow number": "[99999999999999999999999]\nFirst.",
	}
	for name, content := range cases {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "prompts.txt")
			if err := os.WriteFile(path, []byte(content), 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := ReadPrompts(path); err == nil {
				t.Fatal("invalid prompts must fail before generation")
			}
		})
	}
}
