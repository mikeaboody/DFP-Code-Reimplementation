# ZDoom dependencies
sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev

# Boost libraries
sudo apt-get install libboost-all-dev

# Python 2 dependencies
sudo apt-get install python-dev python-pip
pip install numpy
# or install Anaconda 2 and add it to PATH

# Python 3 dependencies
sudo apt-get install python3-dev python3-pip
pip3 install numpy
# or install Anaconda 3 and add it to PATH

# Lua binding dependencies
sudo apt-get install liblua5.1-dev
# Lua shipped with Torch can be used instead, so it isn't needed if installing via LuaRocks
