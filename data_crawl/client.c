#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define IPADDR 	"ssel.sejong.ac.kr"
#define PORT	1863


struct head {
	int cmd;
	int len;
	int ret;
};

struct config {
	int cycle;
	int buf_len;
};

struct syslog_head {
	int syslog_cnt;
};

struct syslog {
	int fac;
	int sev;
	int cnt;
	int stime, etime;
	int len;
	char msg[10000];
};

struct smartd {

};

void do() 
{
	int client_fd;
	int ret, len;
	struct sockaddr_in client_addr;
	char recv_buf[10400];

	client_fd = socket(PF_INET, SOCK_STREAM, 0);

	client_addr.sin_addr.s_addr = inet_addr(IPADDR);
	client_addr.sin_family = AF_INET;
	client_addr.sin_port = htons(PORT);

	ret = connect(client_fd, (struct sockaddr *)&client_addr, sizeof(client_addr));

	if(ret == -1) {
		fprintf(stderr, "Failure Prediction Data Crawl Client : connect error!\n");
		close(client_fd);
		return -1;
	}

	while (1) {
		read();
		write();		
	}

	return 0;
}


int main(int argc, char *argv[]) 
{
	int pid;
	int ret;

	pid = fork();
	if (pid < 0) {
	   fprintf(stderr, "Failure Prediction Data Crawl Client : main fork error!\n");	
	   return 0;
	} else if (pid > 0) { // parent go die
		return 0;
	}

	chdir("/");
	setsid();

	while(1) {
		pid = fork();
		if (pid < 0) {
			fprintf(stderr, "Failure Prediction Data Crawl Client : sub fork error!\n");
			return 0;
		} else if (pid == 0) { // grand child go break;
			break;
		} else if (pid > 0) { // child go wait
			wait(&ret);
		}	
	}

	do();	

	return 0;
}
