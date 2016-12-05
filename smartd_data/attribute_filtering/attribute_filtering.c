#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

void nospace(char *str)
{
    for(;*str;str++)
    {
        if(' ' == *str)
            *str = '_';
    }
}

void checkdir(char *path)
{
    if(access(path, F_OK))
        mkdir(path, 0755);
}

struct check
{
    int id;
    int indata;
    int check;
};

int parser(char *filename)
{
    char readline[6000] = {0, };
    FILE *fp = NULL;

    fp = fopen(filename, "r");

    while(1)
    {
        int i;
        char *pos = readline;
        int len;
        struct check chk[300] = {0, };

        if(-1 == fscanf(fp, "%[^\n]\n", readline))
            break;

        for(i=0; i<5; i++)
        {
            sscanf(pos, "%*[^,]%n", &len);
            pos += len;
        }

        for(i=0; ; i++)
        {
            char data[256] = {0, };

            if(',' == *pos)
            {
                pos++;
                continue;
            }

            sscanf(pos, "%[^,]%n", data, &len);
            pos += len;

            if('\0' == *pos)
                break;
            else if(',' == *pos)
                pos++;
            else
            {
                fprintf(stderr, "format error!\n");
                exit(1);
            }
        }
    }

    fclose(fp);

    return 1;
}

int searchdir(char *path)
{
    struct dirent **dent;
    int ndent = 0;
    int i;
    char oldwd[256] = {0, };
    
    ndent = scandir(path, &dent, NULL, alphasort);
    if(NULL == dent)
        return 0;

    getwd(oldwd);
    chdir(path);

    for(i=0; i<ndent; i++)
    {
        struct stat st;
        char rpath[256] = {0, };

        if(!strcmp(".", dent[i]->d_name) || !strcmp("..", dent[i]->d_name))
            continue;

        lstat(dent[i]->d_name, &st);

        if(S_ISDIR(st.st_mode))
        {
            searchdir(realpath(dent[i]->d_name, NULL));
        }
        else if(S_ISREG(st.st_mode))
            if(parser(realpath(dent[i]->d_name, NULL)))
                printf("%s\n", realpath(dent[i]->d_name, rpath));
    }

    chdir(oldwd);

    return 1;
}

int main(int argc, char *argv[])
{
    if(2 != argc)
    {
        fprintf(stderr, "%s [target driectory] > out file\n", argv[0]);

        return 1;
    }

    searchdir(realpath(argv[1], NULL));

    return 0;
}
