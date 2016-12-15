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
    char name[50];
    int check;
};

struct check g_chk[90];
int g_dircnt = 0;
int g_dirproc = 0;
char g_dirpath[256];

int parser(char *filename)
{
    char readline[6000] = {0, };
    FILE *fp = NULL;
    int is_title = 1;

    fp = fopen(filename, "r");

    while(1)
    {
        int i;
        char *pos = readline;
        int len;


        if(-1 == fscanf(fp, "%[^\n]%*c", readline))
            break;

        for(i=0; i<5; i++)
        {
            sscanf(pos, "%*[^,]%*c%n", &len);
            pos += len;
        }

        for(i=0; ; i++)
        {
            char data[256] = {0, };

            if(',' == *pos)
            {
                pos++;
                g_chk[i].check = 0;
                if('\0' == *(pos+1))
                    g_chk[i+1].check = 0;
                continue;
            }

            sscanf(pos, "%[^,]%n", data, &len);
            pos += len;
            if(is_title)
                sprintf(g_chk[i].name, "%s", data);

            if('\0' == *pos)
            {
                is_title = 0;
                break;
            }
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

    if(0 == g_dircnt)
        g_dircnt = ndent;

    for(i=0; i<ndent; i++)
    {
        struct stat st;
        char rpath[256] = {0, };

        if(!strcmp(".", dent[i]->d_name) || !strcmp("..", dent[i]->d_name))
            continue;

        lstat(dent[i]->d_name, &st);

        if(S_ISDIR(st.st_mode))
        {
            sprintf(g_dirpath, "%s", realpath(dent[i]->d_name, NULL));
            searchdir(realpath(dent[i]->d_name, NULL));
            g_dirproc++;
        }
        else if(S_ISREG(st.st_mode))
            if(parser(realpath(dent[i]->d_name, NULL)))
            {
                printf("[dir: %04d/%04d][file: %05d/%05d][path: %s]\n", g_dircnt, g_dirproc, i, ndent, g_dirpath);
                //printf("%s\n", realpath(dent[i]->d_name, rpath));
            }
    }

    chdir(oldwd);

    return 1;
}

int main(int argc, char *argv[])
{
    int i;
    if(2 != argc)
    {
        fprintf(stderr, "%s [target driectory] > out file\n", argv[0]);

        return 1;
    }

    for(i=0; i<90; i++)
        g_chk[i].check = 1;

    searchdir(realpath(argv[1], NULL));

    for(i=0; i<90; i++)
        if(g_chk[i].check)
            printf("%s\n", g_chk[i].name);

    return 0;
}
