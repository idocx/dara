![DARA logo](logo/dara.jpg)

# DARA: Data-driven automated Rietveld analysis for phase search and refinement

Automated phase search with BGMN.

## Installation

## Install DARA

First, git clone the repository. Then navigate to the directory and pip install:

```bash
cd dara
pip install .
```


### Special instructions: installation on Lawrencium (LBNL)

The supplied BGMNwin folder may not work on your machine/cluster.

For example, installing BGMN on Lawrencium (or perhaps another older Linux cluster) may lead to an error with GLIBC versions when you run bgmn:

    version `GLIBC_2.29 not found (required by …)

To fix, we need to install GLIBC 2.29. Based on:
<https://stackoverflow.com/questions/50564999/lib64-libc-so-6-version-glibc-2-14-not-found-why-am-i-getting-this-error>

```bash
mkdir ~/glibc229
cd ~/glibc229
wget http://ftp.gnu.org/gnu/glibc/glibc-2.29.tar.gz
tar zxvf glibc-2.29.tar.gz
cd glibc-2.29
mkdir build
cd build
../configure --prefix=$HOME/.local
make -j4
make install
```

WARNING: you can’t set LD_LIBRARY_PATH without breaking everything. Instead, we take this approach:

**INSTALL PATCHELF**

Git clone and follow setup instructions here: <https://github.com/NixOS/patchelf?tab=readme-ov-file>

This command will fix every binary. Make sure DARA is installed in a folder called $HOME/dara/

```bash
patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/bgmn/BGMNwin/bgmn && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/bgmn/BGMNwin/eflech && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/bgmn/BGMNwin/geomet && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/bgmn/BGMNwin/gertest && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/bgmn/BGMNwin/index && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/bgmn/BGMNwin/lamtest && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/bgmn/BGMNwin/makegeq && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/bgmn/BGMNwin/output && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/bgmn/BGMNwin/plot1 && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/bgmn/BGMNwin/spacegrp && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/bgmn/BGMNwin/teil && patchelf --set-interpreter $HOME/.local/lib/ld-linux-x86-64.so.2 --set-rpath $HOME/.local/lib/ $HOME/dara/dara/bgmn/BGMNwin/verzerr
```

---
**Note**

IF GLIBC installation error occurs, and it looks related to the version of make/gmake:

    checking version of gmake... 3.82, bad

Then you must install newer version and symbolic link:

```bash
curl -O http://ftp.gnu.org/gnu/make/make-4.2.1.tar.gz
tar xvf make-4.2.1.tar.gz
cd make-4.2.1
./configure --prefix=$HOME/.local/bin && make && make install
export PATH=/$HOME/.local/bin:$PATH
ln -s $HOME/.local/bin/make $HOME/.local/bin/gmake
```

Now try again to install GLIBC….
