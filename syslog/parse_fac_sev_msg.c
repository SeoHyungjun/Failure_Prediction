#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct loginfo
{
    char facility[64];
    char severity[64];
    char msg[512];
};

int parse(char* log, struct loginfo *info)
{
    int len = 0;

    if(NULL == log || '\0' == *log || NULL == info)
        return len;

    sscanf(log, "%[^.].%[^:]:%*[^:]:%*[^:]:%*[^:]:%[^\n]\n%n", info->facility, info->severity, info->msg, &len);

    return len;
}

int main(int argc, char *argv[])
{
    int i;

    while(1)
    {
        char buffer[2024] = {0, };
        struct loginfo parselog;
        int len = 0;

        fscanf(stdin, "%[^\n]%*c", buffer);

        memset(&parselog, 0x00, sizeof(struct loginfo));
        parse(buffer, &parselog);
        printf("fac:%s, sec:%s, msg:%s\n", parselog.facility, parselog.severity, parselog.msg);
    }

    return 0;
}
