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
	"fmt"

	"github.com/NVIDIA/go-gpuallocator/gpuallocator"
)

var alignedAllocationPolicy = gpuallocator.NewBestEffortPolicy() // 一致的分配政策

// alignedAlloc调用为计算首选分配而设置的alignedAllocationPolicy。
func (r *resourceManager) alignedAlloc(available, required []string, size int) ([]string, error) {
	var devices []string

	availableDevices, err := gpuallocator.NewDevicesFrom(available) // 从所有可用的nvvm .Devices中创建一个设备列表。

	if err != nil {
		return nil, fmt.Errorf("unable to retrieve list of available devices: %v", err)
	}

	requiredDevices, err := gpuallocator.NewDevicesFrom(required)
	if err != nil {
		return nil, fmt.Errorf("unable to retrieve list of required devices: %v", err)
	}

	allocatedDevices := alignedAllocationPolicy.Allocate(availableDevices, requiredDevices, size)

	for _, device := range allocatedDevices {
		devices = append(devices, device.UUID)
	}

	return devices, nil
}

// alloc运行一个标准的分配算法来决定哪些设备应该被优先使用。
// 目前，这里没有任何智能操作。我们计划在将来扩展它，以实现更复杂的分配算法。
func (r *resourceManager) alloc(available, required []string, size int) ([]string, error) {
	remainder := r.devices.Subset(available).Difference(r.devices.Subset(required)).GetIDs()
	devices := append(required, remainder...) // 剩余部分
	if len(devices) < size {
		return nil, fmt.Errorf("not enough available devices to satisfy allocation")
	}
	return devices[:size], nil
}

// getPreferredAllocation runs an allocation algorithm over the inputs.
// The algorithm chosen is based both on the incoming set of available devices and various config settings.
func (r *resourceManager) getPreferredAllocation(available, required []string, size int) ([]string, error) {
	// 如果所有可用设备都是没有副本的完整GPU，则计算在这些设备上的对齐分配。
	if !r.Devices().ContainsMigDevices() && // 完整设备
		!AnnotatedIDs(available).AnyHasAnnotations() { // 没有声明
		return r.alignedAlloc(available, required, size)
	}

	// Otherwise, run a standard allocation algorithm.
	return r.alloc(available, required, size)
}
