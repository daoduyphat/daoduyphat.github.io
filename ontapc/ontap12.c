//Tìm số max
#include<stdio.h>
#include<stdlib.h>

int main(){
    system("cls");
    int a,b,c;
    printf("Nhap so A = "); scanf("%d",&a);
    printf("Nhap so B = "); scanf("%d",&b);
    printf("Nhap so C = "); scanf("%d",&c);

    if(a > b){
        if(a > c){
            printf("A = %d is max",a);
        } else{
            printf("C = %d is max",c);
        }
    } else{
         if(b > c){
            printf("B = %d is max",b);
         } else{
            printf("C = %d is max",c);
         }
    }
}