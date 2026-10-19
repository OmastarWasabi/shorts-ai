package main

import (
	"os"
	"testing"
)

func TestReadConfig(t *testing.T) {
	tests := []struct {
		name    string
		env     map[string]string
		want    config
		wantErr bool
	}{
		{name: "defaults", want: config{true, true}},
		{name: "voice only", env: map[string]string{"GENERATE_IMAGES": "false"}, want: config{true, false}},
		{name: "images only", env: map[string]string{"GENERATE_VOICE": "false"}, want: config{false, true}},
		{name: "both off", env: map[string]string{"GENERATE_VOICE": "false", "GENERATE_IMAGES": "false"}},
		{name: "both on", env: map[string]string{"GENERATE_VOICE": "true", "GENERATE_IMAGES": "true"}, want: config{true, true}},
		{name: "voice typo", env: map[string]string{"GENERATE_VOICE": "ture"}, wantErr: true},
		{name: "images typo", env: map[string]string{"GENERATE_IMAGES": "flase"}, wantErr: true},
		{name: "empty is invalid", env: map[string]string{"GENERATE_IMAGES": ""}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, name := range []string{"GENERATE_VOICE", "GENERATE_IMAGES"} {
				t.Setenv(name, "")
				if err := os.Unsetenv(name); err != nil {
					t.Fatal(err)
				}
			}
			for name, value := range tt.env {
				t.Setenv(name, value)
			}
			got, err := readConfig()
			if (err != nil) != tt.wantErr {
				t.Fatalf("readConfig error = %v, want error %v", err, tt.wantErr)
			}
			if !tt.wantErr && got != tt.want {
				t.Fatalf("readConfig = %+v, want %+v", got, tt.want)
			}
		})
	}
}
