# ADAPT-DB
African Dams Project Database

New machine setup:
install Ubuntu:
    follow instructions from http://www.ubuntu.com
    (tested on ubuntu 16.04 LTS)
install TeamViewer:
    download and install the software from:
http://download.teamviewer.com/download/teamviewer_i386.deb
setup teamviewer to start automatically
by-pass the password-protected login:
    go to “system settings”, “user accounts”, press “unlock”, and toggle “automatic login”
Disable screen lock:
    go to “system settings”, “brightness and lock”, and toggle “automatic login”
Tethys installation:
install anaconda3 (a Python distribution with useful packages):
    https://www.continuum.io/downloads
    add anaconda to path. Paste the following at the end of ~/.profile:
        # add anaconda to PATH
PATH="$HOME/anaconda3/bin:$PATH"
install MySQL Server:
e.g. from the Ubuntu Software Center
alternative: http://www.cyberciti.biz/faq/howto-install-mysql-on-ubuntu-linux-16-04/
    sudo apt update
    sudo apt upgrade
    sudo apt install mysql-server mysql-client
install MySQL Workbench
e.g. from the Ubuntu Software Center
install the libmysqlclient-dev package:
    e.g. from the Ubuntu Software Center
    alternative: type in the command line:
sudo apt-get install libmysqlclient-dev
install the mysqlclient MySQL driver for Python:
    cd {anaconda install folder}/bin
    ./pip install mysqlclient
install django:
cd {anaconda install folder}/bin
    ./pip install django
create a database for django:
    on MySQL workbench run:
        CREATE DATABASE adaptdb_data CHARACTER SET utf8;
    create a django user:
        Use the workbench. Users and Privileges / Add Account:
            Login Name: django
            Limit to Hosts: localhost
            Password: {to be matched in django’s settings.py}
            Schema Privileges: {add all privileges to the ‘adaptdb_data’ schema}
install django countries:
cd {anaconda install folder}/bin
./pip install django-countries
install OpenCL for GPU:
    read https://wiki.tiker.net/OpenCLHowTo for background information
    download the latest GPU drivers (e.g. https://goo.gl/66BVxr, but this depends on your GPU vendor)
Install OpenCL for CPU (Intel i7, on Ubuntu 16.04):
    download the intel OpenCL drivers from http://goo.gl/xQL1Ws
    download from http://packages.ubuntu.com/trusty/lsb-core
    download from http://packages.ubuntu.com/trusty/lsb-security
    install lsb:
        cd ~/Downloads
sudo dpkg -i lsb-core_4.1+Debian11ubuntu6_amd64.deb
        sudo apt-get update
        sudo apt-get install -f
sudo dpkg -i lsb-security_4.1+Debian11ubuntu6_amd64.deb
sudo apt-get update
sudo apt-get upgrade
Install the intel drivers:
    extract opencl_runtime_16.1_x64_ubuntu_5.2.0.10002.tgz
    cd ~/Downloads/opencl_runtime_16.1_x64_ubuntu_5.2.0.10002
    sudo bash ./install_GUI.sh
install PyOpenCL:
    download pyopencl from https://pypi.python.org/pypi/pyopencl
follow the instructions in: https://wiki.tiker.net/PyOpenCL/Installation/Linux
pip install mako
python configure.py \
  --cl-inc-dir=/opt/intel/opencl/include \
  --cl-lib-dir=/opt/intel/opencl \
  --cl-libname=OpenCL
sudo make install
install redis:
    follow instructions in http://redis.io/topics/quickstart:
install celery:
cd {anaconda install folder}/bin
    ./pip install celery
    ./pip install django-celery
install parallel:
    sudo apt-get install parallel
create a Django user with appropriate privileges:
    from the root direction of the server (where manage.py is) run:
        {anaconda install folder}/bin/python manage.py createsuperuser
install wsgi:
    sudo apt-get install libapache2-mod-wsgi-py3
    sudo service apache2 restart
Install HDF5:
    download code from https://www.hdfgroup.org/HDF5/release/obtainsrc.html
    extract and launch a terminal in the created folder
    run:
        ./configure --prefix=/usr/local/hdf5 --enable-fortran --enable-cxx --enable-hl --enable-shared
        make
                make check
                sudo make install
                sudo make check-install
Install netCDF4:
    download code from ftp://ftp.unidata.ucar.edu/pub/netcdf/ 
    extract and launch a terminal in the created folder
    run:
        NCDIR=/usr/local
H5DIR=/usr/local/hdf5
    CPPFLAGS=-I${H5DIR}/include LDFLAGS=-L${H5DIR}/lib ./configure --prefix=${NCDIR} --enable-netcdf-4 --enable-shared
    make check
    sudo make install
        pip install netcdf4
To edit the code:
Install Java:
    type in the command line:
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install openjdk-8-jre openjdk-8-jdk

install Eclipse:
    download from: https://eclipse.org
    unzip to a desired folder
    create shortcut: http://askubuntu.com/questions/13758
        paste .desktop file into ~/.local/share/applications/ and edit it.
install the PyDev module:
    http://www.pydev.org
reference anaconda as the interpreter (in Eclipse: Window / Preferences / PyDev / Interpreters / Python Interpreter / New…
name it as anaconda. Point towards {anaconda install}/bin/python
setup cython:
    cd ~/eclipse/plugins/org.python.pydev_{PyDev version}/pysrc
    python setup_cython.py build_ext --inplace
fix scroll in Eclipse:
    Preferences / PyDev / Editor / Overview Ruler Minimap / Show vertical scrollbar
To prepare the web server for use:
install no-ip:
    go to https://www.noip.com/
    sign up and choose domain name (e.g. adaptdbzra.ddns.net)
    Install application (https://goo.gl/GRKvrt) 
install apache http:
    https://help.ubuntu.com/lts/serverguide/httpd.html

setup apache http:
    .
secure mysql server:
    type in the command line “sudo mysql_secure_installation”
http://unix.stackexchange.com/questions/16890/how-to-make-a-machine-accessible-from-the-lan-using-its-hostname
To run the server:
start redis:
    {redis install folder}/src/redis-server
start celery:
    cd ~/Tethys
celery -A tethys worker -l info --pool=solo
start django:
    cd ~/Tethys
python manage.py runserver --insecure 0.0.0.0:8000
Troubleshooting:
WebGL disabled in Firefox:
    go to about:support. check for error causes in the graphics table.
    if “GLContext is disabled due to a previous crash” is a cause go to about:config.
    Try changing “webgl.force-enabled” to true and “gfx.crash-guard.status.glcontext” to 0.

Notes:
geoJSONs:
    their names must be trivial
    the coordinates should be WGS84
