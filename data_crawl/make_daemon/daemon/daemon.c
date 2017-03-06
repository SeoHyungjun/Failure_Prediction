#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <syslog.h>

void daemon_main()
{
	int i = 0;
	FILE *fp = fopen("/var/log/fpd.log", "w+");
	
	while (1) {
		i++;
		sleep(2);
		fprintf(fp, "Failure Prediction logging... %d...\n", i);
		fflush(fp);
	}

	fclose(fp);
}

extern int proc();
int main(int argc, char *argv[]) 
{
	pid_t pid, sid;
	FILE *pid_file;
	int ret;

	pid = fork();
	if (pid < 0) {
		fprintf(stderr, "Failure Prediction Data Crawl Client : main fork error!\n");	
		exit(EXIT_FAILURE);
	} else if (pid > 0) { // parent go die
		exit(EXIT_SUCCESS);
	}

	umask(0);

	openlog(argv[0], LOG_NOWAIT|LOG_PID, LOG_USER);

	syslog(LOG_NOTICE, "Sucessfully started Failure Prediction Data Crawl Daemon!\n");

	sid = setsid();
	if (sid < 0) {
		syslog(LOG_ERR, "Could not create process group\n");
		exit(EXIT_FAILURE);
	}

	pid_file = fopen("/var/run/fpd.pid", "w");
	fprintf(pid_file, "%d\n", getpid());
	fclose(pid_file);
	chmod("/var/run/fpd.pid", 0644);

	if((chdir("/")) < 0) {
		syslog(LOG_ERR, "Could not change working directory to /\n");
		exit(EXIT_FAILURE);
	}

	close(STDIN_FILENO);
	close(STDOUT_FILENO);
	close(STDERR_FILENO);
	
	proc();

	//daemon_main();	
	
	closelog();
	return EXIT_SUCCESS;
}


