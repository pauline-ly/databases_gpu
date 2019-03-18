#!/bin/bash 
#
#test linked
#
set -x
set -u

FILE=my_idea_10_12__.txt
tf=megakv
cd build 
cmake ..
make
cd ..

echo "\033[1;31;46m Starting test \033[0m \n"
echo "file: ${tf}" >>${FILE}

./build/${tf}  >${FILE}

testf ()
{

    COUNTER=100000
#    10K -100K
    while [ ${COUNTER} -le 100000000 ]
    do
      echo ">>>\n\n\n\ntest numbers: ${COUNTER} " >>${FILE}
      echo ">>>test numbers: ${COUNTER} " >>${FILE}
      echo ">>>test numbers: ${COUNTER} " >>${FILE}
      
      ./build/${tf} ${COUNTER}  >>${FILE}

        
      COUNTER=$((COUNTER * 2))
    done
}

#testf

pwd
