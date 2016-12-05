#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

void checkdir(char *path)
{
    if(access(path, F_OK))
        mkdir(path, 0755);
}

struct append_data
{
    int id;
    char name[20];
    int index;
} g_append_list[] = 
{
    {22, "smart_22_nomalized", 31},
    {22, "smart_22_raw", 32},

    {220, "smart_220_nomalized", 67},
    {220, "smart_220_raw", 68},

    {222, "smart_222_nomalized", 69},
    {222, "smart_222_raw", 70},

    {224, "smart_224_nomalized", 73},
    {224, "smart_224_raw", 74},

    {226, "smart_226_nomalized", 77},
    {226, "smart_226_raw", 78},
};

int parser(char *filename, char *out)
{
    FILE *fp = fopen(filename, "r");
    FILE *out_fp = NULL;
    char oldwd[256] = {0, };
    char out_filename[256] = {0, };
    int is_title = 1;

    if(!fp)
        return 0;

    checkdir(out);

    getwd(oldwd);
    chdir(out);

    sprintf(out_filename, "%s/%s", out, filename);
    out_fp = fopen(out_filename, "w");

    while(1)
    {
        char readline[6000] = {0, };
        char outline[6000] = {0, };
        char *pos = readline;
        char *out_pos = outline;
        int i;
        int len = 0;
        int list_index = 0;

        if(-1 == fscanf(fp, "%[^\n]\n", readline))
            break;

        for(i=0; ; i++)
        {
            char value[256] = {0, };

            if(i == g_append_list[list_index].index)
            {
                if(is_title)
                    out_pos += sprintf(out_pos, "%s,", g_append_list[list_index].name);
                else
                    out_pos += sprintf(out_pos, ",");

                list_index++;
                continue;
            }

            if(',' == *pos)
            {
                sprintf(out_pos, ",");
                pos++;
                out_pos++;
                continue;
            }

            sscanf(pos, "%[^,]%n", value, &len);
            sprintf(out_pos, "%s", value);
            pos += len;
            out_pos += len;

            if(',' == *pos)
            {
                sprintf(out_pos, ",");
                pos++;
                out_pos++;
            }
            else if('\0' == *pos)
            {
                is_title = 0;
                fprintf(out_fp, "%s\n", outline);
                break;
            }
            else
            {
                fprintf(stderr, "csv file formatted error\n");

                exit(1);
            }
        }

    }

    chdir(oldwd);

    fclose(fp);
    fclose(out_fp);
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

