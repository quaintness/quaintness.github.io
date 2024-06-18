# Linux tutorial

This note are taken from the [freeCodeCamp.org](https://www.youtube.com/watch?v=sWbUDq4S6Y8)

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

In general, executable programs and scripts should live in the `/bin` or `/usr/bin` or `/sbin` or `/usr/sbiin` or `/usr/local/bin` or `/usr/local/sbin` or  somewhere under `/opt` or  an user account space, such as `/home/username/bin`

- One way to locate the programs is to use the `which` utility.
- `whereis` is a good alternative of `which` when it can not find the program, because it look for a broader range of system directories.

**Accessing directories**

Terminal's default directory is `/home` directory.

One can print the exact path by typing

```shell
echo $HOME
```

|    Command     | Result                                                       |
| :------------: | ------------------------------------------------------------ |
|     `pwd`      | Displays the present directory                               |
| `cd ~` or `cd` | Change to your home directory (shortcut name is ~ (tilde))   |
|    `cd ..`     | Change to parent directoty (`..`)                            |
|     `cd -`     | Change to previous directory (`-` (minus))                   |
|    `pushd`     | Saves the current working directory in memory (via a directory stack) so it can be returned to at any time, places the new filepath at the top of the stack, and changes to the new filepath. |
|     `popd`     | returns to the path at the top of the directory stack. This directory stack is accessed by the command `dirs` in Unix or `Get-Location -stack` in Windows PowerShell. |

:warning: You can use the combination `pushd [pathname]` :arrow_right: `popd` :arrow_right: `cd -` to do a fast return.

**Absolute and relative paths**

![image-20230708161023144](https://raw.githubusercontent.com/OctopussGarden/ImageRepo/main/imgs/image-20230708161023144.png)

- `.` present directory
- `..` parent directory
- `~` your home directory

**Exploring the filesystem**

| Command | Usage                                                        |
| :-----: | ------------------------------------------------------------ |
| `cd /`  | Change your current working directory to root (/) directory or path you supply |
|  `la`   | list of contents of the present working directory            |
| `ls -a` | List of all files, including hidden files and directories (those who named start with `.`) |
| `tree`  | Displays a tree view of the filesystem                       |

**Hard links**

```shell
quaint@jarvis:~/Desktop$ touch file1
quaint@jarvis:~/Desktop$ ln file1 file2
quaint@jarvis:~/Desktop$ ls -li file?  # `-li` prints the node number (which is unique) in the 1st col
303920 -rw-rw-r-- 2 quaint quaint 0  7月  8 18:17 file1
303920 -rw-rw-r-- 2 quaint quaint 0  7月  8 18:17 file2
```

By create hard link, you can have multiple file names for one single file. (In this case, `file1`)

**Soft (Symbolic) links**

Created with `-s` option. 

```shell
quaint@jarvis:~/Desktop$ ln -s file1 file3
quaint@jarvis:~/Desktop$ ls -li file1 file3 #file3 is not a regular file
303920 -rw-rw-r-- 2 quaint quaint 0  7月  8 18:17 file1
304063 lrwxrwxrwx 1 quaint quaint 5  7月  8 18:42 file3 -> file1 # file3 has a different enote number `304063`
```

<u>*Tips:*</u>

- Soft links are convenient cause it can easily be modified to point to different places (e.g. create ==shortcuts== to a location).
- Soft links can point to objects even on different locations. if the object doesn't exist, it is a dangly link.

### Working with files

|   Command   | Function                                                     |
| :---------: | ------------------------------------------------------------ |
|    `wc`     | Word count                                                   |
| `cat [-n]`  | View file contents [with line number]                        |
| `less [-N]` | View file with page view (scroll with *space bar*) [with line number] |
|    `tac`    | Print the entire file backwards                              |

- `touch` resets the file's time stamp to match the current time.

  `touch <filename>`  create a empty file as a placeholder, alternatively, you can use `echo > <filename>`

  `touch -t <timestamp> <filename>` can set the date and time stamp of the give file to a specific value. *e.g. `touch -t 201804301015 somefile`*

- `mkdir` and `rmdir`

  `mkdir <dirname>`  create a directory

  `rmdir`

  `rm -rf` 

- Moving, renaming or removing a file

  | Command | Usage                       |
  | :-----: | --------------------------- |
  |  `mv`   | Rename a file               |
  |  `rm`   | Remove a file               |
  | `rm -f` | Forcefully remove a file    |
  | `rm -i` | Interactively remove a file |

- Moving, renaming or removing a directory

  | Command  | Usage                                     |
  | :------: | ----------------------------------------- |
  |   `mv`   | Rename a file                             |
  | `rmdir`  | Remove an empty directory                 |
  | `rm -rf` | Forcefully remove a directory recursively |

- Modifying the command prompt

  Can modify command's `PS1` value to change its default behavior.

- Standard file stream

  <center>File descriptor</center>

![image-20230710161355713](https://raw.githubusercontent.com/OctopussGarden/ImageRepo/main/imgs/image-20230710161355713.png)

- I/O Redirection `>` `<`

  ```shell
  do_something < input-file #`input-file` is the input data that can be consumed for program `do_something`
  do_something >  output-file #Write outputs to a file
  do_something 2> error-file
  ```

  A special shorthand notation can send anything written to file descriptor 2(standard error) to the same place  as file descriptor 1 (standard out). Using the following command:

  ```shell
  do_something > all-output-file 2>&1
  do_something >& all-output-file #bash's alternative command (easier)
  ```

- Pipes `|`

  You can pipe one program's output into a new program as its input. As following:

  ```shell
  command1 | command2 | command3 # Pipeline example
  ```

  Pros: 

  1. **Saving spaces**: No need to store temporary data
  2. **More efficient**: Reducing reading and writing from disk

- Searching for files

  - `locate`

    Take advantage of a pre-constructed file database, can use `grep` to filter and constrain output list.

    *`grep` print only the lines that contain one or more specified string.*
    
    ```shell
    locate zip | grep bin #This command lists all files and directories with both `zip` and `bin` in their name
    ```
    

## Regular expression

- Interactive Tutorial : [Rexone](https://regexone.com/)

- [Language Guide for Python](https://regexone.com/references/python)

  > 1. **Raw Python Strings**: Using raw Python strings (i.e. `r"strings"`), which is easier to read, instead regular Python strings
  >
  > 2. ##### **Matching a string**: `re` package
  >
  >    ```python
  >    matchObject = re.search(pattern, input_str, flags=0)
  >    ```
  >
  > 3. **Capturing groups**:
  >
  >    ```python
  >    # perform a global search over the whole input string, return a list
  >    matchList = re.findall(pattern, input_str, flags=0)
  >    # returns an iterator of re.MatchObjects to walk through
  >    matchList = re.finditer(pattern, input_str, flags=0)
  >    ```
  >
  > 4. **Finding and replacing strings**
  >
  >    ```python
  >    replacedString = re.sub(pattern, replacement_pattern, input_str, count, flags=0)
  >    ```
  >
  > 5. **`re` Flags**
  >
  >    - [`re.IGNORECASE`](https://docs.python.org/3.6/library/re.html#re.IGNORECASE) makes the pattern case insensitive so that it matches strings of different capitalizations
  >    - [`re.MULTILINE`](https://docs.python.org/3.6/library/re.html#re.MULTILINE) is necessary if your input string has newline characters (*\n*), this flag allows the start and end metacharacter (*^* and *$* respectively) to match at the beginning and end of each line instead of at the beginning and end of the whole input string
  >    - [`re.DOTALL`](https://docs.python.org/3.6/library/re.html#re.DOTALL) allows the dot (*.*) metacharacter match all characters, including the newline character (*\n*)
  >
  > 6. **Compiling a pattern for performance**
  >
  >    ```python
  >    regexObject = re.compile(pattern, flags=0)
  >    ```
  >
  > ***Links***
  >
  > - [Python Documentation for Regular Expressions](https://docs.python.org/3.6/library/re.html)
  >
  > - [Python Compatible Regex Tester](https://regex101.com/#python)
