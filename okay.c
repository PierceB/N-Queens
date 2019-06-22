#include <stdio.h>

int main(){
	unsigned int a = 4;
	unsigned int b = 5;
	printf("a %p\n", &a);
	printf("b %p\n", &b);
	printf("%p\n", ((void*)&b - (void*)&a));
	return 1;
}
