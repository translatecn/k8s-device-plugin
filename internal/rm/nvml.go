/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY Type, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package rm

import (
	"bytes"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/NVIDIA/go-nvml/pkg/dl"
	"github.com/NVIDIA/go-nvml/pkg/nvml"

	"github.com/NVIDIA/k8s-device-plugin/internal/mig"
)

const (
	nvmlXidCriticalError = nvml.EventTypeXidCriticalError
)

// nvmlDevice wraps an nvml.Device with more functions.
type nvmlDevice nvml.Device

// nvmlEvent holds relevant data about an NVML Event.
type nvmlEvent struct {
	UUID              *string
	GpuInstanceID     *uint
	ComputeInstanceID *uint
	Etype             uint64
	Edata             uint64
}

// nvmlLookupSymbol checks to see if the given symbol is present in the NVMl library.
func nvmlLookupSymbol(symbol string) error {
	lib := dl.New("libnvidia-ml.so.1", dl.RTLD_LAZY|dl.RTLD_GLOBAL)
	if lib == nil {
		return fmt.Errorf("error instantiating DynamicLibrary for NVML")
	}
	err := lib.Open()
	if err != nil {
		return fmt.Errorf("error opening DynamicLibrary for NVML: %v", err)
	}
	defer lib.Close()
	return lib.Lookup(symbol)
}

// nvmlWaitForEvent waits for an NVML Event
func nvmlWaitForEvent(es nvml.EventSet, timeout uint) (nvmlEvent, error) {
	data, ret := es.Wait(uint32(timeout))
	if ret != nvml.SUCCESS {
		return nvmlEvent{}, fmt.Errorf("%v", nvml.ErrorString(ret))
	}

	uuid, ret := data.Device.GetUUID()
	if ret != nvml.SUCCESS {
		return nvmlEvent{}, fmt.Errorf("%v", nvml.ErrorString(ret))
	}

	isMig, ret := data.Device.IsMigDeviceHandle()
	if ret != nvml.SUCCESS {
		return nvmlEvent{}, fmt.Errorf("%v", nvml.ErrorString(ret))
	}

	if !isMig {
		data.GpuInstanceId = 0xFFFFFFFF
		data.ComputeInstanceId = 0xFFFFFFFF
	}

	event := nvmlEvent{
		UUID:              &uuid,
		Etype:             uint64(data.EventType),
		Edata:             uint64(data.EventData),
		GpuInstanceID:     uintPtr(data.GpuInstanceId),
		ComputeInstanceID: uintPtr(data.ComputeInstanceId),
	}

	return event, nil
}

// nvmlRegisterEventForDevice registers an Event for a device with a specific UUID.
func nvmlRegisterEventForDevice(es nvml.EventSet, event int, uuid string) error {
	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("%v", nvml.ErrorString(ret))
	}

	for i := 0; i < count; i++ {
		d, ret := nvml.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			return fmt.Errorf("%v", nvml.ErrorString(ret))
		}

		duuid, ret := d.GetUUID()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("%v", nvml.ErrorString(ret))
		}

		if duuid != uuid {
			continue
		}

		ret = d.RegisterEvents(uint64(event), es)
		if ret != nvml.SUCCESS {
			return fmt.Errorf("%v", nvml.ErrorString(ret))
		}

		return nil
	}

	return fmt.Errorf("nvml: device not found")
}

// walkMigProfiles walks all of the possible MIG profiles across all GPU devices reported by NVML
func walkMigProfiles(f func(p string) error) error {
	visited := make(map[string]bool)
	return walkGPUDevices(func(i int, gpu nvml.Device) error {
		capable, err := nvmlDevice(gpu).isMigCapable()
		if err != nil {
			return fmt.Errorf("error checking if GPU %v is MIG capable: %v", i, err)
		}
		if !capable {
			return nil
		}
		for i := 0; i < nvml.GPU_INSTANCE_PROFILE_COUNT; i++ {
			giProfileInfo, ret := gpu.GetGpuInstanceProfileInfo(i)
			if ret == nvml.ERROR_NOT_SUPPORTED {
				continue
			}
			if ret == nvml.ERROR_INVALID_ARGUMENT {
				continue
			}
			if ret != nvml.SUCCESS {
				return fmt.Errorf("error getting GPU instance profile info for '%v': %v", i, nvml.ErrorString(ret))
			}

			g := giProfileInfo.SliceCount
			gb := ((giProfileInfo.MemorySizeMB + 1024 - 1) / 1024)
			p := fmt.Sprintf("%dg.%dgb", g, gb)

			if visited[p] {
				continue
			}

			err := f(p)
			if err != nil {
				return err
			}

			visited[p] = true
		}
		return nil
	})
}

// walkMigDevices walks all of the MIG devices across all GPU devices reported by NVML
func walkMigDevices(f func(i, j int, d nvml.Device) error) error {
	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("error getting GPU device count: %v", nvml.ErrorString(ret))
	}

	for i := 0; i < count; i++ {
		device, ret := nvml.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting device handle for GPU with index '%v': %v", i, nvml.ErrorString(ret))
		}

		migEnabled, err := nvmlDevice(device).isMigEnabled()
		if err != nil {
			return fmt.Errorf("error checking if MIG is enabled on GPU with index '%v': %v", i, err)
		}

		if !migEnabled {
			continue
		}

		err = nvmlDevice(device).walkMigDevices(func(j int, device nvml.Device) error {
			return f(i, j, device)
		})
		if err != nil {
			return fmt.Errorf("error walking MIG devices on GPU with index '%v': %v", i, err)
		}
	}
	return nil
}

// walkMigDevices walks all of the MIG devices on a specific GPU device reported by NVML
func (d nvmlDevice) walkMigDevices(f func(i int, d nvml.Device) error) error {
	count, ret := nvml.Device(d).GetMaxMigDeviceCount()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("error getting max MIG device count: %v", nvml.ErrorString(ret))
	}

	for i := 0; i < count; i++ {
		device, ret := nvml.Device(d).GetMigDeviceHandleByIndex(i)
		if ret == nvml.ERROR_NOT_FOUND {
			continue
		}
		if ret == nvml.ERROR_INVALID_ARGUMENT {
			continue
		}
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting MIG device handle at index '%v': %v", i, nvml.ErrorString(ret))
		}
		err := f(i, device)
		if err != nil {
			return err
		}
	}
	return nil
}

// isMigCapable checks if a device is MIG capable or not
func (d nvmlDevice) isMigCapable() (bool, error) {
	err := nvmlLookupSymbol("nvmlDeviceGetMigMode")
	if err != nil {
		return false, nil
	}

	_, _, ret := nvml.Device(d).GetMigMode()
	if ret == nvml.ERROR_NOT_SUPPORTED {
		return false, nil
	}
	if ret != nvml.SUCCESS {
		return false, fmt.Errorf("error getting MIG mode: %v", nvml.ErrorString(ret))
	}

	return true, nil
}

// getMigProfile gets the MIG profile name associated with the given MIG device
func (d nvmlDevice) getMigProfile() (string, error) {
	isMig, err := d.isMigDevice()
	if err != nil {
		return "", fmt.Errorf("error checking if device is a MIG device: %v", err)
	}
	if !isMig {
		return "", fmt.Errorf("device handle is not a MIG device")
	}

	attr, ret := nvml.Device(d).GetAttributes()
	if ret != nvml.SUCCESS {
		return "", fmt.Errorf("error getting MIG device attributes: %v", nvml.ErrorString(ret))
	}

	g := attr.GpuInstanceSliceCount
	c := attr.ComputeInstanceSliceCount
	gb := ((attr.MemorySizeMB + 1024 - 1) / 1024)

	var p string
	if g == c {
		p = fmt.Sprintf("%dg.%dgb", g, gb)
	} else {
		p = fmt.Sprintf("%dc.%dg.%dgb", c, g, gb)
	}

	return p, nil
}

// getPaths returns the set of Paths associated with the given device (MIG or GPU)
func (d nvmlDevice) getPaths() ([]string, error) {
	isMig, err := d.isMigDevice()
	if err != nil {
		return nil, fmt.Errorf("error checking if device is a MIG device: %v", err)
	}

	if !isMig {
		minor, ret := nvml.Device(d).GetMinorNumber()
		if ret != nvml.SUCCESS {
			return nil, fmt.Errorf("error getting GPU device minor number: %v", nvml.ErrorString(ret))
		}
		return []string{fmt.Sprintf("/dev/nvidia%d", minor)}, nil
	}

	uuid, ret := nvml.Device(d).GetUUID()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting UUID of MIG device: %v", nvml.ErrorString(ret))
	}

	paths, err := mig.GetMigDeviceNodePaths(uuid)
	if err != nil {
		return nil, fmt.Errorf("error getting MIG device paths: %v", err)
	}

	return paths, nil
}

// getNumaNode returns the NUMA node associated with the given device (MIG or GPU)
func (d nvmlDevice) getNumaNode() (*int, error) {
	isMig, err := d.isMigDevice()
	if err != nil {
		return nil, fmt.Errorf("error checking if device is a MIG device: %v", err)
	}

	if isMig {
		parent, ret := nvml.Device(d).GetDeviceHandleFromMigDeviceHandle()
		if ret != nvml.SUCCESS {
			return nil, fmt.Errorf("error getting parent GPU device from MIG device: %v", nvml.ErrorString(ret))
		}
		d = nvmlDevice(parent)
	}

	info, ret := nvml.Device(d).GetPciInfo()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting PCI Bus Info of device: %v", nvml.ErrorString(ret))
	}

	// Discard leading zeros.
	busID := strings.ToLower(strings.TrimPrefix(int8Slice(info.BusId[:]).String(), "0000"))

	b, err := os.ReadFile(fmt.Sprintf("/sys/bus/pci/devices/%s/numa_node", busID))
	if err != nil {
		// Report nil if NUMA support isn't enabled
		return nil, nil
	}

	node, err := strconv.ParseInt(string(bytes.TrimSpace(b)), 10, 8)
	if err != nil {
		return nil, fmt.Errorf("eror parsing value for NUMA node: %v", err)
	}

	if node < 0 {
		return nil, nil
	}

	n := int(node)
	return &n, nil
}

// walkGPUDevices walks all of the GPU devices reported by NVML
func walkGPUDevices(f func(i int, d nvml.Device) error) error {
	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("error getting device count: %v", nvml.ErrorString(ret))
	}

	for i := 0; i < count; i++ {
		device, ret := nvml.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting device handle for index '%v': %v", i, nvml.ErrorString(ret))
		}
		err := f(i, device)
		if err != nil {
			return err
		}
	}
	return nil
}

// isMigEnabled checks if MIG is enabled on the given GPU device
func (d nvmlDevice) isMigEnabled() (bool, error) {
	err := nvmlLookupSymbol("nvmlDeviceGetMigMode")
	if err != nil {
		return false, nil
	}

	mode, _, ret := nvml.Device(d).GetMigMode()
	if ret == nvml.ERROR_NOT_SUPPORTED {
		return false, nil
	}
	if ret != nvml.SUCCESS {
		return false, fmt.Errorf("error getting MIG mode: %v", nvml.ErrorString(ret))
	}

	return (mode == nvml.DEVICE_MIG_ENABLE), nil
}

// isMigDevice checks if the given NVML device is a MIG device (as opposed to a GPU device)
func (d nvmlDevice) isMigDevice() (bool, error) {
	err := nvmlLookupSymbol("nvmlDeviceIsMigDeviceHandle")
	if err != nil {
		return false, nil
	}
	isMig, ret := nvml.Device(d).IsMigDeviceHandle()
	if ret != nvml.SUCCESS {
		return false, fmt.Errorf("%v", nvml.ErrorString(ret))
	}
	return isMig, nil
}

// nvmlNewEventSet creates a new NVML EventSet
func nvmlNewEventSet() nvml.EventSet {
	set, _ := nvml.EventSetCreate()
	return set
}

// nvmlDeleteEventSet deletes an NVML EventSet
func nvmlDeleteEventSet(es nvml.EventSet) {
	es.Free()
}
