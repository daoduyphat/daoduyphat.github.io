/*
    Exercise 1: (5 điểm)

    1. (3 điểm) Viết hàm trong ngôn ngữ C để nhận vào một chuỗi ký tự làm tham số;
       sau đó trích xuất tất cả các số nguyên có trong chuỗi và tính tổng các số này.
       Ví dụ:
       Input chuỗi: "abc12def34"
       Output: "The numbers are 12 and 34. The sum is 46"

    2. (2 điểm) Viết một hàm để tính tổng các số nguyên từ 1 đến N: 1 + 2 + 3 + ... + N
*/

#include <stdio.h>
#include <ctype.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

// Hàm trích xuất và tính tổng các số trong chuỗi
int sum_of_numbers(const char *str) {
    int sum = 0;
    int num = 0;
    bool is_number = false;

    printf("The numbers are ");
    for (int i = 0; str[i] != '\0'; i++) {
        if (isdigit(str[i])) {
            num = num * 10 + (str[i] - '0');
            is_number = true;
        } else {
            if (is_number) {
                printf("%d ", num);
                sum += num;
                num = 0;
                is_number = false;
            }
        }
    }

    if (is_number) {
        printf("%d ", num);
        sum += num;
    }

    printf(". The sum is %d\n", sum);
    return sum;
}

// Hàm tính tổng từ 1 đến N
int sum_series(int N) {
    int sum = 0;
    for (int i = 1; i <= N; i++) {
        sum += i;
    }
    return sum;
}

int main() {
    system("cls"); 
    char input[100];
    printf("Enter a string: ");
    fgets(input, sizeof(input), stdin);

    // Xóa ký tự newline nếu có
    input[strcspn(input, "\n")] = '\0';

    sum_of_numbers(input);

    int N;
    printf("\nEnter N to calculate sum from 1 to N: ");
    scanf("%d", &N);
    int total = sum_series(N);
    printf("Sum from 1 to %d is: %d\n", N, total);

    return 0;
}
