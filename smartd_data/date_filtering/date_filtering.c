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

int parser(char *filename, char *sdate, char *edate)
{
    char cmd[512] = {0, };
    char date[100] = {0, };
    FILE *fp = NULL;

    sprintf(cmd, "cat %s | grep %s", filename, sdate);

    fp = popen(cmd, "r");
    if(!fp)
        return 0;

    if(-1 == fscanf(fp, "%[^,]", date))
    {
        pclose(fp);

        return 0;
    }

    pclose(fp);

    if(strcmp(sdate, date))
        return 0;

    sprintf(cmd, "cat %s | grep %s", filename, edate);

    fp = popen(cmd, "r");
    if(!fp)
        return 0;

    if(-1 == fscanf(fp, "%[^,]", date))
    {
        pclose(fp);

        return 0;
    }

    if(strcmp(edate, date))
        return 0;

    return 1;
}

int searchdir(char *path, char *sdate, char *edate)
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
            searchdir(realpath(dent[i]->d_name, NULL), sdate, edate);
        }
        else if(S_ISREG(st.st_mode))
            if(parser(realpath(dent[i]->d_name, NULL), sdate, edate))
                printf("%s\n", realpath(dent[i]->d_name, rpath));
    }

    chdir(oldwd);

    return 1;
}

int main(int argc, char *argv[])
{
    if(4 != argc)
    {
        fprintf(stderr, "%s [target driectory] [start date] [end date] > out file\n", argv[0]);

        return 1;
    }

    searchdir(realpath(argv[1], NULL), argv[2], argv[3]);

    return 0;
}
