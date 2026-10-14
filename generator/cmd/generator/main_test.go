package main

import (
	"os"
	"testing"
)

func TestValidateInputs(t *testing.T) {
	previous, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(t.TempDir()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(previous); err != nil {
			t.Error(err)
		}
	})
	if err := validateInputs(); err == nil {
		t.Fatal("missing input files must fail")
	}
	if err := os.WriteFile("story.txt", []byte("Обычный текст."), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile("prompts.txt", []byte("[01]\nA scene.\n[01]\nDuplicate."), 0600); err != nil {
		t.Fatal(err)
	}
	if err := validateInputs(); err == nil {
		t.Fatal("invalid prompts must fail before either generator starts")
	}
	if err := os.WriteFile("prompts.txt", []byte("[01]\nA scene."), 0600); err != nil {
		t.Fatal(err)
	}
	if err := validateInputs(); err != nil {
		t.Fatal(err)
	}
}
