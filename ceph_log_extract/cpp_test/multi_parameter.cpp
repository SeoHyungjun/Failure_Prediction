#include <stdio.h>
#include <ctype.h>

#define test printf("hi\n");   \
             printf("hello\n");

class A{
    public:
        char const a = 'c';
};

int main() {
//    A test;
//    printf("%c\n", test.a);
    test
    return 0;
}
