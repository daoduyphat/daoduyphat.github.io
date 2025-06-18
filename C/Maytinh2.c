// Cũng là máy tính nhưng cách khác
#include <stdio.h>      // Thư viện cho nhập xuất chuẩn
#include <stdlib.h>     // Thư viện cho các hàm tiện ích, như system()

int main(void) {
    system("cls");      // Xóa màn hình console (chỉ hoạt động trên Windows)
    char op;            // Biến lưu phép toán (+, -, *, /)
    double first, second;   // Hai biến lưu hai số thực nhập vào
    printf("Nhap so thu nhat: ");     // Hiển thị yêu cầu nhập số thứ nhất
    scanf("%lf", &first);             // Đọc số thực đầu tiên từ bàn phím
    printf("Nhap phep toan (+, -, *, /): "); // Hiển thị yêu cầu nhập phép toán
    scanf(" %c", &op);                        // Đọc ký tự phép toán (có khoảng trắng để bỏ qua ký tự xuống dòng)
    printf("Nhap so thu hai: ");      // Hiển thị yêu cầu nhập số thứ hai
    scanf("%lf", &second);            // Đọc số thực thứ hai từ bàn phím
    switch (op) {                     // Kiểm tra phép toán vừa nhập
        case '+':
            printf("%.1lf + %.1lf = %.1lf", first, second, first + second); // Cộng hai số và in kết quả
            break;
        case '-':
            printf("%.1lf - %.1lf = %.1lf", first, second, first - second); // Trừ hai số và in kết quả
            break;
        case '*':
            printf("%.1lf * %.1lf = %.1lf", first, second, first * second); // Nhân hai số và in kết quả
            break;
        case '/':
            printf("%.1lf / %.1lf = %.1lf", first, second, first / second); // Chia hai số và in kết quả
            break;
        default:
            printf("Phep toan khong hop le"); // Nếu nhập sai phép toán thì báo lỗi
    }
    return 0;   // Kết thúc chương trình
}