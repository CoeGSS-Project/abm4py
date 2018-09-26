#/bin/bash
rstfile=${1-'dakota'}
csvfile=${2-'dakota'}
dakota_restart_util to_tabular "${rstfile}.rst" dakota.tabular
sed -r 's/[ ]+/,/g' dakota.tabular > "${csvfile}.csv"
rm dakota.tabular
