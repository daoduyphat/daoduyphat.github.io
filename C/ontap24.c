//Tính toán với hai số nguyên
// Viết chương trình C để tính tổng, hiệu, tích, thương của hai số nguyên
#include <stdio.h>
int add(int a, int b){
    int sum = a + b;
    return sum; // lệnh return trả về giá trị có cùng kiểu dữ liệu với kiểu dữ liệu của hàm
}
// Hàm tính hiệu của a - b
int subtract(int a, int b){
    return a - b;
}
// Hàm tính tích a * b
int mutiply(int a, int b){
    return a * b;
}
// Hàm tính thương của a / b
// Lưu ý cần != 0
// phải ép kiểu tử hoặc mẫu để thu được kết quả là số thực 
float divide(int a, int b){
    return (float) a / b;
}
void main(void){
    int a = 3, b = 4;
    // Sử dụng tên hàm để gọi nó, 
    // truyền các tham số đúng theo kiểu của hàm yêu cầu
    printf("\n%d + %d = %d", a, b, add(a, b)); 
    printf("\n%d - %d = %d", a, b, subtract(a, b));
    printf("\n%d * %d = %d", a, b, mutiply(a, b));
    if(b != 0){
        printf("\n%d / %d = %f", a, b, divide(a, b));
    }
}