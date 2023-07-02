/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"syscall"
	"time"

	"github.com/NVIDIA/gpu-monitoring-tools/bindings/go/nvml"
	spec "github.com/NVIDIA/k8s-device-plugin/api/config/v1"
	"github.com/NVIDIA/k8s-device-plugin/internal/rm"
	"github.com/fsnotify/fsnotify"
	cli "github.com/urfave/cli/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

var version string // This should be set at build time to indicate the actual version

func main() {
	var configFile string

	c := cli.NewApp()
	c.Version = version
	c.Action = func(ctx *cli.Context) error {
		return start(ctx, c.Flags)
	}

	c.Flags = []cli.Flag{
		&cli.StringFlag{
			Name:  "mig-strategy",
			Value: spec.MigStrategyNone,
			// MIG模式定义了如何将GPU划分为多个较小的实例，以及每个实例所拥有的资源数量。
			// 通常有两种MIG模式可供选择：静态模式和动态模式。静态模式需要在系统启动时进行配置，而动态模式允许在运行时根据需要进行实例的创建和销毁。
			Usage:   "在支持MIG的gpu上公开MIG设备所需的策略:\n\t\t[none | single | mixed]",
			EnvVars: []string{"MIG_STRATEGY"},
		},
		&cli.BoolFlag{
			Name:    "fail-on-init-error",
			Value:   true,
			Usage:   "如果在初始化过程中遇到错误，则失败插件，否则将无限期阻塞",
			EnvVars: []string{"FAIL_ON_INIT_ERROR"},
		},
		&cli.StringFlag{
			Name:    "nvidia-driver-root",
			Value:   "/",
			Usage:   "NVIDIA驱动安装的根路径 (typical values are '/' or '/run/nvidia/driver')",
			EnvVars: []string{"NVIDIA_DRIVER_ROOT"},
		},
		&cli.BoolFlag{
			Name:    "pass-device-specs",
			Value:   false,
			Usage:   "通过Allocate()将deviceSpec列表传递给kubelet",
			EnvVars: []string{"PASS_DEVICE_SPECS"},
		},
		&cli.StringFlag{
			Name:    "device-list-strategy",
			Value:   spec.DeviceListStrategyEnvvar,
			Usage:   "将设备列表传递到底层运行时所需的策略:\n\t\t[envvar | volume-mounts | cdi-annotations]",
			EnvVars: []string{"DEVICE_LIST_STRATEGY"},
		},
		&cli.StringFlag{
			Name:    "device-id-strategy",
			Value:   spec.DeviceIDStrategyUUID,
			Usage:   "将设备id传递到底层运行时所需的策略:\n\t\t[uuid | index]",
			EnvVars: []string{"DEVICE_ID_STRATEGY"},
		},
		&cli.StringFlag{
			Name:        "config-file",
			Usage:       "配置文件的路径，作为命令行选项或环境变量的替代方案",
			Destination: &configFile,
			EnvVars:     []string{"CONFIG_FILE"},
		},
	}

	err := c.Run(os.Args)
	if err != nil {
		log.SetOutput(os.Stderr)
		log.Printf("Error: %v", err)
		os.Exit(1)
	}
}

func start(c *cli.Context, flags []cli.Flag) error {
	log.Println("Starting FS watcher.")
	watcher, err := newFSWatcher(pluginapi.DevicePluginPath)
	if err != nil {
		return fmt.Errorf("failed to create FS watcher: %v", err)
	}
	defer watcher.Close()

	log.Println("Starting OS watcher.")
	sigs := newOSWatcher(syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)

	var restarting bool
	var restartTimeout <-chan time.Time
	var plugins []*NvidiaDevicePlugin
restart:
	// If we are restarting, stop plugins from previous run.
	if restarting {
		err := stopPlugins(plugins)
		if err != nil {
			return fmt.Errorf("error stopping plugins from previous run: %v", err)
		}
	}

	log.Println("Starting Plugins.")
	plugins, restartPlugins, err := startPlugins(c, flags, restarting)
	if err != nil {
		return fmt.Errorf("error starting plugins: %v", err)
	}

	if restartPlugins {
		log.Printf("Failed to start one or more plugins. Retrying in 30s...")
		restartTimeout = time.After(30 * time.Second)
	}

	restarting = true

	// Start an infinite loop, waiting for several indicators to either log
	// some messages, trigger a restart of the plugins, or exit the program.
	for {
		select {
		// If the restart timout has expired, then restart the plugins
		case <-restartTimeout:
			goto restart

		// Detect a kubelet restart by watching for a newly created
		// 'pluginapi.KubeletSocket' file. When this occurs, restart this loop,
		// restarting all of the plugins in the process.
		case event := <-watcher.Events:
			if event.Name == pluginapi.KubeletSocket && event.Op&fsnotify.Create == fsnotify.Create {
				log.Printf("inotify: %s created, restarting.", pluginapi.KubeletSocket)
				goto restart
			}

		// Watch for any other fs errors and log them.
		case err := <-watcher.Errors:
			log.Printf("inotify: %s", err)

		// Watch for any signals from the OS. On SIGHUP, restart this loop,
		// restarting all of the plugins in the process. On all other
		// signals, exit the loop and exit the program.
		case s := <-sigs:
			switch s {
			case syscall.SIGHUP:
				log.Println("Received SIGHUP, restarting.")
				goto restart
			default:
				log.Printf("Received signal \"%v\", shutting down.", s)
				goto exit
			}
		}
	}
exit:
	err = stopPlugins(plugins)
	if err != nil {
		return fmt.Errorf("error stopping plugins: %v", err)
	}
	return nil
}

func stopPlugins(plugins []*NvidiaDevicePlugin) error {
	log.Println("Stopping plugins.")
	for _, p := range plugins {
		p.Stop()
	}
	log.Println("Shutting down NVML.")
	if err := nvml.Shutdown(); err != nil {
		return fmt.Errorf("error shutting down NVML: %v", err)
	}
	return nil
}

// disableResourceRenamingInConfig 暂时禁用插件的资源重命名功能。
// 我们计划在未来的发布中重新启用此功能。
func disableResourceRenamingInConfig(config *spec.Config) {
	// Disable resource renaming through config.Resource
	if len(config.Resources.GPUs) > 0 || len(config.Resources.MIGs) > 0 {
		log.Printf("Customizing the 'resources' field is not yet supported in the config. Ignoring...")
	}
	config.Resources.GPUs = nil
	config.Resources.MIGs = nil

	// Disable renaming / device selection in Sharing.TimeSlicing.Resources
	renameByDefault := config.Sharing.TimeSlicing.RenameByDefault
	setsNonDefaultRename := false
	setsDevices := false
	for i, r := range config.Sharing.TimeSlicing.Resources {
		if !renameByDefault && r.Rename != "" {
			setsNonDefaultRename = true
			config.Sharing.TimeSlicing.Resources[i].Rename = ""
		}
		if renameByDefault && r.Rename != r.Name.DefaultSharedRename() {
			setsNonDefaultRename = true
			config.Sharing.TimeSlicing.Resources[i].Rename = r.Name.DefaultSharedRename()
		}
		if !r.Devices.All {
			setsDevices = true
			config.Sharing.TimeSlicing.Resources[i].Devices.All = true
			config.Sharing.TimeSlicing.Resources[i].Devices.Count = 0
			config.Sharing.TimeSlicing.Resources[i].Devices.List = nil
		}
	}
	if setsNonDefaultRename {
		log.Printf("Setting the 'rename' field in sharing.timeSlicing.resources is not yet supported in the config. Ignoring...")
	}
	if setsDevices {
		log.Printf("Customizing the 'devices' field in sharing.timeSlicing.resources is not yet supported in the config. Ignoring...")
	}
}

func validateFlags(config *spec.Config) error {
	if *config.Flags.Plugin.DeviceListStrategy != spec.DeviceListStrategyEnvvar && *config.Flags.Plugin.DeviceListStrategy != spec.DeviceListStrategyVolumeMounts {
		return fmt.Errorf("invalid --device-list-strategy option: %v", *config.Flags.Plugin.DeviceListStrategy)
	}

	if *config.Flags.Plugin.DeviceIDStrategy != spec.DeviceIDStrategyUUID && *config.Flags.Plugin.DeviceIDStrategy != spec.DeviceIDStrategyIndex {
		return fmt.Errorf("invalid --device-id-strategy option: %v", *config.Flags.Plugin.DeviceIDStrategy)
	}
	return nil
}

func loadConfig(c *cli.Context, flags []cli.Flag) (*spec.Config, error) {
	config, err := spec.NewConfig(c, flags)
	if err != nil {
		return nil, fmt.Errorf("unable to finalize config: %v", err)
	}

	err = validateFlags(config)
	if err != nil {
		return nil, fmt.Errorf("unable to validate flags: %v", err)
	}
	config.Flags.GFD = nil
	return config, nil
}

func startPlugins(c *cli.Context, flags []cli.Flag, restarting bool) ([]*NvidiaDevicePlugin, bool, error) {
	// Load the configuration file
	log.Println("Loading configuration.")
	config, err := loadConfig(c, flags)
	if err != nil {
		return nil, false, fmt.Errorf("unable to load config: %v", err)
	}
	disableResourceRenamingInConfig(config)

	// Start NVML
	log.Println("Initializing NVML.")
	if err := nvml.Init(); err != nil {
		log.SetOutput(os.Stderr)
		log.Printf("Failed to initialize NVML: %v.", err)
		log.Printf("If this is a GPU node, did you set the docker default runtime to `nvidia`?")
		log.Printf("You can check the prerequisites at: https://github.com/NVIDIA/k8s-device-plugin#prerequisites")
		log.Printf("You can learn how to set the runtime at: https://github.com/NVIDIA/k8s-device-plugin#quick-start")
		log.Printf("If this is not a GPU node, you should set up a toleration or nodeSelector to only deploy this plugin on GPU nodes")
		log.SetOutput(os.Stdout)
		if *config.Flags.FailOnInitError {
			return nil, false, fmt.Errorf("failed to initialize NVML: %v", err)
		}
		select {}
	}

	// Update the configuration file with default resources.
	log.Println("Updating config with default resource matching patterns.")
	err = rm.AddDefaultResourcesToConfig(config)
	if err != nil {
		return nil, false, fmt.Errorf("unable to add default resources to config: %v", err)
	}

	// Print the config to the output.
	configJSON, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return nil, false, fmt.Errorf("failed to marshal config to JSON: %v", err)
	}
	log.Printf("\nRunning with config:\n%v", string(configJSON))

	// Get the set of plugins.
	log.Println("Retreiving plugins.")
	migStrategy, err := NewMigStrategy(config)
	if err != nil {
		return nil, false, fmt.Errorf("error creating MIG strategy: %v", err)
	}
	plugins := migStrategy.GetPlugins()

	// Loop through all plugins, starting them if they have any devices
	// to serve. If even one plugin fails to start properly, try
	// starting them all again.
	started := 0
	for _, p := range plugins {
		// Just continue if there are no devices to serve for plugin p.
		if len(p.Devices()) == 0 {
			continue
		}

		// Start the gRPC server for plugin p and connect it with the kubelet.
		if err := p.Start(); err != nil {
			log.SetOutput(os.Stderr)
			log.Println("Could not contact Kubelet. Did you enable the device plugin feature gate?")
			log.Printf("You can check the prerequisites at: https://github.com/NVIDIA/k8s-device-plugin#prerequisites")
			log.Printf("You can learn how to set the runtime at: https://github.com/NVIDIA/k8s-device-plugin#quick-start")
			log.SetOutput(os.Stdout)
			return plugins, true, nil
		}
		started++
	}

	if started == 0 {
		log.Println("No devices found. Waiting indefinitely.")
	}

	return plugins, false, nil
}
