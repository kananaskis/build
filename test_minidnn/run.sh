if [ ! -d /data/test/ ];then
   mkdir -p /data/test/
fi
cd /data/test/
chmod 777 *
export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/apex/com.android.runtime/lib64:$LD_LIBRARY_PATH
./testModelDebug
./testModelRelease
