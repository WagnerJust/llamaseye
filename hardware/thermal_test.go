package hardware

import (
	"context"
	"testing"
	"time"
)

func TestWaitCool_Disabled(t *testing.T) {
	tm := &ThermalMonitor{
		HW:       &HardwareInfo{},
		Disabled: true,
	}
	// Should return immediately
	done := make(chan struct{})
	go func() {
		tm.WaitCool(context.Background())
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Error("WaitCool with Disabled=true did not return immediately")
	}
}

func TestWaitCool_NoCmds(t *testing.T) {
	// No temp commands = temp reads return 0, always below limits
	tm := &ThermalMonitor{
		HW:          &HardwareInfo{CPUTempCmd: "", GPUTempCmd: ""},
		CPULimit:    88,
		GPULimit:    81,
		PollSeconds: 1,
	}
	done := make(chan struct{})
	go func() {
		tm.WaitCool(context.Background())
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(200 * time.Millisecond):
		t.Error("WaitCool with no temp cmds did not return immediately")
	}
}

func TestWaitCool_ContextCancel(t *testing.T) {
	// Use a command that will make temps appear high (by returning invalid output)
	// but the context cancels quickly
	tm := &ThermalMonitor{
		HW: &HardwareInfo{
			CPUTempCmd: "echo 999", // will appear as high temp
		},
		CPULimit:    88,
		GPULimit:    81,
		PollSeconds: 60,
	}
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	done := make(chan struct{})
	go func() {
		tm.WaitCool(ctx)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
		t.Error("WaitCool did not respect context cancellation")
	}
}
