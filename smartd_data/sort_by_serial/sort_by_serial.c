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

int parser(char *filename, char *out)
{
    FILE *fp = fopen(filename, "r");
    char oldwd[256] = {0, };
    char title[6000] = {0, };

    if(!fp)
        return 0;

    checkdir(out);

    getwd(oldwd);
    chdir(out);

    fscanf(fp, "%[^\n]\n", title); //delete title

    while(1)
    {
        int ret = 0;
        char date[100] = {0 ,};
        char serial[100] = {0, };
        char model[100] = {0, };
        char etc[4096] = {0, };
        FILE *out_fp = NULL;
        char out_name[120] = {0, };

        ret = fscanf(fp, "%[^,],%[^,],%[^,],%[^\n]\n", date, serial, model, etc);
        if(-1 == ret)
            break;

        nospace(model);
        checkdir(model);
        chdir(model);

        nospace(serial);

        sprintf(out_name, "%s.csv", serial);
        if(!access(out_name, 0))
            out_fp = fopen(out_name, "a");
        else
        {
            out_fp = fopen(out_name, "w");
            fprintf(out_fp, "%s\n",title);
        }

        fprintf(out_fp, "%s,%s,%s,%s\n", date, serial, model, etc);
        fclose(out_fp);
        chdir("..");

        //printf("date : %s, model : %s, serial : %s\n"
        //        "etc : %s\n", date, serial, model, etc);
    }

    chdir(oldwd);

    fclose(fp);
}

int searchdir(char *path, char *out)
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
        if(!strcmp(".", dent[i]->d_name) || !strcmp("..", dent[i]->d_name))
            continue;

        parser(dent[i]->d_name, out);
        printf("%04d/%04d\n", ndent, i+1);
    }

    chdir(oldwd);

    return 1;
}

int main(int argc, char *argv[])
{
    char out_rpath[256] = {0, }; 
    if(3 != argc)
    {
        fprintf(stderr, "%s [csv directory] [out directory]\n", argv[0]);

        return 1;
    }

    realpath(argv[2], out_rpath);

    searchdir(argv[1], out_rpath);

    return 0;
}
