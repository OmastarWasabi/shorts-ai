package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadEnv(t *testing.T) {
	const loadedKey = "HISTORY_GLITCH_TEST_LOADED"
	const existingKey = "HISTORY_GLITCH_TEST_EXISTING"
	previous, exists := os.LookupEnv(loadedKey)
	if err := os.Unsetenv(loadedKey); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if exists {
			os.Setenv(loadedKey, previous)
		} else {
			os.Unsetenv(loadedKey)
		}
	})
	t.Setenv(existingKey, "from-environment")
	path := filepath.Join(t.TempDir(), ".env")
	content := "# Settings\n\n" + loadedKey + " = \"value=with-equals\"\n" + existingKey + "=from-file\n"
	if err := os.WriteFile(path, []byte(content), 0600); err != nil {
		t.Fatal(err)
	}
	if err := loadEnv(path); err != nil {
		t.Fatal(err)
	}
	if os.Getenv(loadedKey) != "value=with-equals" {
		t.Fatal("value was not loaded from .env")
	}
	if os.Getenv(existingKey) != "from-environment" {
		t.Fatal(".env must not replace an existing environment variable")
	}
}

func TestLoadEnvMissingFile(t *testing.T) {
	if err := loadEnv(filepath.Join(t.TempDir(), ".env")); err != nil {
		t.Fatal(err)
	}
}

func TestLoadEnvErrorsDoNotExposeValues(t *testing.T) {
	for _, content := range []string{"secret-without-equals", "BAD KEY=secret", "TEST_KEY=\"secret"} {
		path := filepath.Join(t.TempDir(), ".env")
		if err := os.WriteFile(path, []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
		err := loadEnv(path)
		if err == nil || strings.Contains(err.Error(), "secret") {
			t.Fatalf("expected error without the secret value: %v", err)
		}
	}
}
