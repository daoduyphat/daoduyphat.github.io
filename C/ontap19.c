//In ra cách đọc số có hai chữ số
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
int main(void)
    {
        system("cls");
        //khai báo biến n và yêu cầu
        //người dùng nhập vào từ bàn phím
        int n;
        printf("\nNhập vào số n: ");
        scanf("%d", &n);
        //a là đơn vị
        int a = n % 10;
        //b là chục
        int b = n / 10;
        //điều kiện số có hai chữ số
        if(n<10 || n > 99){
            printf("Phai nhap vao so co hai chu so!! ");
        }
        else{
            //in ra hàng chục
            switch(b){
                case 1: printf("Mười ");break;
                case 2: printf("Hai mươi ");break;
                case 3: printf("Ba mươi ");break;
                case 4: printf("Bốn mươi ");break;
                case 5: printf("Năm mươi ");break;
                case 6: printf("Sáu mươi ");break;
                case 7: printf("Bảy mươi ");break;
                case 8: printf("Tám mươi ");break;
                case 9: printf("Chín mươi ");break;
            }
            //in ra hàng đơn vị
            switch(a){
                case 1: printf("một ");break;
                case 2: printf("hai ");break;
                case 3: printf("ba ");break;
                case 4: printf("bốn ");break;
                case 5: printf("lăm ");break;
                case 6: printf("sáu ");break;
                case 7: printf("bảy ");break;
                case 8: printf("tám ");break;
                case 9: printf("chín ");break;
            }
        }
        printf("\n------------------------------------\n");
    return 0;
}