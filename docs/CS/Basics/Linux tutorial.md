# Linux tutorial

This note are taken from the [freeCodeCamp.org](https://www.youtube.com/watch?v=sWbUDq4S6Y8&t=8614s)

[TOC]

## Chap 2 Linux Philosophy and Concepts

**Basic terms**

- Kernel (Glue links the hardwares and applications)
- Distribution(distros) (Group of applications)
- Boot loader (Start OS)
- Service (Program run as a background process)
- File system (Method for storing and organizing file)
- X Windows system (Provide standard toolkits and protocols to build GUI)
- Desktop environment (GUI on top of OS)
- Command Line (Interface for typing commands)
- Shell (Command Line Interpreter)

==Linux Distribution== = A kernel + other software tools

### Chap 2 summary

- Linux borrows from Unix
- Linux accesses features through files and file-like objects
- Linux is a multitasking multi-user OS
- Linux distro includes kernel and tools

## Chap 3 Linux Basics and System Startup

### ==**System startup process**==

`Power on` :arrow_right: `BIOS` :arrow_right: `MBR` :arrow_right: `Boot Loader ` :arrow_right: `Kernel (Linux OS)` :arrow_right: `Initial RAM disk - initramfs image` :arrow_right: `/sbin/init (parent process)`

- `BIOS` (*Basic Input and Output Systems*) = POST (*Power on self-test*)

  - BIOS software is stored on a ROM chip on the motherboard
  - After BIOS, remainder of the boot process is controlled by the OS

- `MBR` is abbreviation for `Master Boot Record`

  - MBR also known as First Sector of Hard Disk

- `Bootloader` is stored on hard disks in the system, either in the boot sector (traditional BIOS or MBR) or EFI partition or unified extensible firmware interfaces(UEFI)

  - Most common one: GRand Unified Bootloader (GRUB)/ ISO Linux (<u>Boot from removable media</u>)/DOS u-boot (Boot on embedded devices appliances)

  - When booting Linux, the bootloader is responsible for loading the kernel image and the initial RAM disk or file system into memory

  - Bootloader has 2 distinct stages

    First stage Bootloader

    1. For systems using BIOS MBR method

       Bootloader resides at the MBR (Size of MBR is 512 bytes), bootloader examines the partition table and finds a bootable partition. 

       After that, bootloader searches for the second stage Bootloader (e.g. GRUB) and loads it into RAM.

    2. For systems using EFI/UEFI method

       UEFI firmware reads its boot manager data to determine which UEFI application is to be launched and from where, then firmware launches the UEFI application (e.g. GRUB)

    Second stage Bootloader (Resides under `\boot`)

    Allow user to choose which OS to boot, after choosing, the bootloader loads the <u>kernel</u> of the selected OS into RAM and passes control to it.

    Then, kernel usually uncompress itself first and then check, analyze hardwares and initialize any hardwares built into the kernel.

  - `Initial RAM disk - initramfs image`

    Contains programs and binary files that perform all actions needed to mount the proper root file system.

    ![image-20230701184305426](https://raw.githubusercontent.com/OctopussGarden/ImageRepo/main/imgs/image-20230701184305426.png)

    Bootloader loads the kernel and initial file system into memory so that it can be directly used by the kernel. Then kernel initializes and configures the computer's memory, configures all hardwares attached, loads some necessary user space applications.
    
  - `/sbin/init`
  
    After kernel's initial setting, the kernel runs `/sbin/init/`, which is a initial process to start other process to get the system running.
  
    `../init/`is responsible for starting the system, keeping the system running and shutting it down cleanly. Besides that, `../init/`is responsible for acting as a manager when necessary for all non-kernel processes.
  
    - `systemd` takes over `../init/` process.
    
      `/lib/systemd/systemd` using aggressive parallelization techniques instead of sequential serialized set of steps, starts systems faster than `../init/`
    
      one `systemd` command `systemctl` is used for most basic tasks.
  
  - Linux Filesystems Basics
  
    Filesystem is the embodiment of a method storing and organizing arbitrary collections of data in a human usable form. 
  
    ![Screenshot from 2023-07-04 16-20-53](https://raw.githubusercontent.com/OctopussGarden/ImageRepo/main/imgs/Screenshot%20from%202023-07-04%2016-20-53.png)
  
    ![Screenshot from 2023-07-04 18-46-52](https://raw.githubusercontent.com/OctopussGarden/ImageRepo/main/imgs/Screenshot%20from%202023-07-04%2018-46-52.png)
  
  Linux use Filesystem Hierarchy Standard (FHS) to organize its system file. 
  
  

## Chapter 7 Command Line Operations

### Command Line advantages

- No GUI overhead is incurred,
- Virtually any and every task can be accomplished while sitting at the command line.
- Capable of implementing scripts for often-used (or easy-to-forget) tasks and series of procedures.
- Can sign into remote machines anywhere on the Internet.
- Can initiate graphical applications directly from the command line instead of hunting through menus.

### Some basic utilities

 For file operation:

- `cat` used to type out a file (or combine files)

- `head` used to show the first few lines of a file

- `tail` used to show the last few lines of a file

- `man` used to view documentation

- By stopping GUI, one can use `sudo systemctl stop gdm` or `sudo telinit 3`

  To restart GUI service, using `sudo systemctl start gdm` or `sudo telinit 5`

### Basic operations

- `cd`
- `cat`
- `echo`
- `ls`
- `rmdir`
- `man`
- `exit`
- `login`
- `mkdir`

**Login and logout**

After login your own machine, you can also connect and log into remote systems by using SSH.

**Reboot and shut down the system**

Preferred method to shut down the system is to use the `shutdown` command.

- `halt` and `poweroff` issue`shutdown -h` to halt the system.
- `reboot` issue `shutdown -r` to reboot the machine.

Tips:

- `reboot` and `shutdown` from cmd require super user or root access

- Command to notify all users prior shutdown

  ```shell
  shutdown -h 10:00 "Shutting down for scheduled maintenance"
  ```

**Locating applications**

