#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string.h>

/**
 * no_space
 * "a bc  d   ef"
 * s^^c
 * "a bc  d   ef"
 *  s^c
 * "a bc  d   ef"
 *  s^^c
 * "ab c  d   ef"
 *  s^ ^c
 * "ab c  d   ef"
 *   s^^c
 * "abc   d   ef"
 *   s^ ^c
 * "abc   d   ef"
 *    s^^c
 * "abc   d   ef"
 *    s^ ^c
 * "abc   d   ef"
 *    s^  ^c
 * "abcd      ef"
 *    s^   ^c
 * "abcd      ef"
 *     s^   ^c
 * "abcd      ef"
 *     s^    ^c
 * "abcd      ef"
 *     s^     ^c
 * "abcde      f"
 *     s^      ^c
 * "abcde      f"
 *      s^     ^c
 * "abcdef      "
 *      s^      ^c<NUL
 * "abcdef      "
 *       s^     ^c<NUL
 * "abcdef"
 *       s^<NUL ^c<NUL
 * **/

char* erase_element(char *str, char e)
{
    char *ret = str;
    char *ch = str+1;

    while(*str)
    {
        if(e == *str)
        {
            *str = *ch;
            *ch++ = e;
        }
        else
            str++;
    }

    return ret;
}

#define no_space(str) erase_element(str, ' ')
char* no_chs(char *str, char *chs)
{
    for(; *chs; chs++)
        erase_element(str, *chs);
    return str;
}

int main(int argc, char *argv[])
{
    while(1)
    {
        char str[500] = {0, };
        char check_str[500] = {0, };

        if(-1 == fscanf(stdin, "%[^\n]%*c", str))
            break;

        memcpy(check_str, str, sizeof(str));

        no_space(check_str);

        if(2 < strlen(check_str))
            printf("%s\n", str);
    }

    return 0;
}
