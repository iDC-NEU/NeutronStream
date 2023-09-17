#!/bin/bash
#!/bin/bash
######
#usage ./test.sh [dir]
#目前统计的C/C++工程的代码行数（除去了空行）
#要统计其他语言的工程代码行数只需要修改extens的值即可
######
extens=(".c" ".cc" ".cpp" ".h" ".hpp")
filesCount=0
linesCount=0
function is_extens()
{
    for i in ${extens[@]}
    do
        if [ "$i" = "$1" ];then
            return 10
        fi
    done
    return 9
}
function funCount()
{
    for file in `ls $1`
    do
        if [ -d $1"/"$file ];then
            #echo "$1/$file"
            funCount $1"/"$file
        else
            fileName=$1"/"$file
            EXTENSION=".${fileName##*.}"
            is_extens $EXTENSION
            if [ $? -eq 10 ]; then
                lines=`grep -v "^$" $fileName | wc -l`  
                echo "$fileName $lines"
                let linesCount=$linesCount+$lines
                let filesCount=$filesCount+1
            fi
        fi
    done
}
if [ $# -gt 0 ];then
    for m_dir in $@
    do
        m_len=`expr ${#m_dir} - 1`
        if [ ${m_dir:0-1} = "/" ]; then
            funCount ${m_dir:0:m_len}
        else
            funCount $m_dir
        fi
    done
else
    funCount "."
fi
echo "filesCount = $filesCount"
echo "linesCount = $linesCount"