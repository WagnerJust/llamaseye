// Package output handles all file-writing for a sweep: JSONL, Markdown, logs, hardware.json.
package output

import (
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

// Logger writes timestamped messages to stderr and optionally to a log file.
type Logger struct {
	mu      sync.Mutex
	logFile *os.File
	Debug   bool // when true, Debugf writes [DEBUG] lines; when false, Debugf is a no-op
}

// NewLogger creates a Logger. If logPath is non-empty, it opens/creates the file for appending.
func NewLogger(logPath string) (*Logger, error) {
	l := &Logger{}
	if logPath != "" {
		f, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			return nil, err
		}
		l.logFile = f
	}
	return l, nil
}

// Close closes the underlying log file, if any.
func (l *Logger) Close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.logFile != nil {
		_ = l.logFile.Close()
	}
}

// Log writes a timestamped message to stderr (and log file if open).
func (l *Logger) Log(format string, args ...any) {
	ts := time.Now().UTC().Format("2006-01-02T15:04:05Z")
	msg := fmt.Sprintf("[S"+ts+"] "+format, args...)
	l.write(msg)
}

// Warn writes a [WARN]-prefixed message.
func (l *Logger) Warn(format string, args ...any) {
	l.Log("[WARN] "+format, args...)
}

// Debugf writes a [DEBUG]-prefixed message when l.Debug is true; no-op otherwise.
func (l *Logger) Debugf(format string, args ...any) {
	if l.Debug {
		l.Log("[DEBUG] "+format, args...)
	}
}

func (l *Logger) write(msg string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	line := msg + "\n"
	_, _ = io.WriteString(os.Stderr, line)
	if l.logFile != nil {
		_, _ = io.WriteString(l.logFile, line)
	}
}
