#include <stdio.h>

int main()
{
    int num, count, sum = 0;

    printf("Nhap so N can tinh: ");
    scanf("%d", &num);

    // Vong lap de tinh tong
    for(count = 1; count <= num; ++count)
    {
        sum += count; // sum = sum + count;
    }

    printf("Tong = %d", sum);

    return 0;
}
// Chương trình này tính tổng các số từ 1 đến N, trong đó N được nhập từ bàn phím.
// Nó sử dụng vòng lặp for để lặp qua các số từ 1 đến N và cộng dồn vào biến sum.