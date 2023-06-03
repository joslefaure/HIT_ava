## Data Preparation

### Easy Version

1. Download the tar.gz file from [[here]](https://pan.baidu.com/s/1UrflK4IgiVbVBOP5fDHdKA) with code `q5v5`. 

2. run following commands to unzip the file and create a 
symbolic link to the extracted files.

    ```bash
    tar zxvf AVA_compress.tar.gz -C /some/path/
    cd /path/to/HIT_ava/
    mkdir data
    ln -s /some/path/AVA data/AVA
    ```