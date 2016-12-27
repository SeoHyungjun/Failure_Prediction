# !/bin/sh
#
# fpd Failure Prediction data crawling client daemon
#
# chkconfig: # # #
# processname : fpd
#
#

DAEMON=../daemon/fpd
SCRIPT=fpd
SERVICE=fpd


echo "Install Failure Prediction data crawling client daemon..."

if [ -f "/etc/init.d/$SERVICE" ]; then # installed check
	echo "Error: service '$SERVICE' already installed"
	exit 1
fi

if [ `id -u` -ne 0 ]; then # permission check
	echo "Error: user has not permission"
	exit 1
fi


if [ `grep -c 'CentOS' /etc/issue` -eq 1 ]; then #OS check
	UPDATE_RC="chkconfig"
    . /etc/init.d/functions
elif [ `grep -c 'Ubuntu' /etc/issue` -eq 1 ]; then
	UPDATE_RC="update-rc.d"
else
	echo "Error: fpd support only centos or ubuntu"
	exit 1
fi


if [ ! -w /etc/init.d ]; then
	echo "Error: You don't gave me enough permissions to install service"
	exit
else
	echo "0. cp \"$DAEMON\" \"/usr/bin/$SERVICE\"" 		# mv daemon to /usr/bin/
	cp -v "$DAEMON" "/usr/bin/$SERVICE"

	echo "1. cp \"$SCRIPT\" \"/etc/init.d/$SERVICE\"" 	# mv service shell script to init.d 
	cp -v "$SCRIPT" "/etc/init.d/$SERVICE"

	echo "2. touch \"/var/log/$SERVICE.log\"" 			# touch log file
	touch "/var/log/$SERVICE.log"

	if [ $UPDATE_RC = "chkconfig" ]; then # CentOS
		echo "3. \"$UPDATE_RC\" --add \"$SERVICE\" && \"$UPDATE_RC\" --level 2345 \"$SERVICE\" on"
		"$UPDATE_RC" --add "$SERVICE" && "$UPDATE_RC" --level 23455 "$SERVICE" on
	else # Ubuntu
		echo "3. \"$UPDATE_RC\" \"$SERVICE\" defaults"
		"$UPDATE_RC" "$SERVICE" defaults
	fi

	echo "4. service \"$SERVICE\" start"
	service "$SERVICE" start
fi

echo "Complete..."







