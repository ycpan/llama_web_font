ps aux | grep wenda | awk -F ' ' '{print $2}'| xargs kill -9
