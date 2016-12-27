#include <stdio.h>
#include <stdlib.h>

#include <netdb.h>
#include <netinet/in.h>

#include <string.h>

int connection(char *ip, int port)
{
    int sock;
    struct sockaddr_in saddr = {0, };

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if(-1 == sock)
        return -1;
    saddr.sin_family = AF_INET;
    saddr.sin_addr.s_addr = inet_addr(ip);
    saddr.sin_port = htons(port);

    if(-1 == connect(sock, (struct sockaddr*)&saddr, sizeof(saddr)))
        return -1;

    return sock;
}

struct loginfo
{
    char fac[64];
    char sev[64];
    char msg[2048];
};

int main()
{
    int sock;
    FILE *logfd = NULL;
/*
    if(3 != argc)
    {
        fprintf(stderr, "%s [server ip] [port]\n", argv[0]);
        return 1;
    }

    sock = connection(argv[1], atoi(argv[2]));
*/

	sock = connection("210.107.232.93", 3140);
    if(-1 == sock)
        return 1;

    logfd = popen("tail -f /var/log/messages -n 1", "r");
    if(NULL == logfd)
    {
        perror("popen error : ");
        return 1;
    }

    fscanf(logfd, "%*[^\n]%*c");

    while(1)
    {
        char buffer[4096] = {0, };
        struct loginfo parselog = {0, };
        int len = 0;

        if(-1 == fscanf(logfd, "%[^\n]%*c", buffer))
            break;

        printf("%s\n", buffer);

        //sscanf(buffer, "%[^.].%[^:]:%*[^:]:%*[^:]:%*[^:]:%[^\n]\n%n", parselog.fac, parselog.sev, parselog.msg, &len);
        write(sock, buffer, strlen(buffer));
    }

    pclose(logfd);

    close(sock);

    return 0;
}
