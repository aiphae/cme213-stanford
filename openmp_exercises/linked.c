#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#ifndef N
#define N 5
#endif
#ifndef FS
#define FS 38
#endif

typedef struct node {
    int data;
    int fib_data;
    struct node *next;
} node;

int fib(int n) {
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);
}

void process_work(node *p) {
    int n = p->data;
    p->fib_data = fib(n);
}

node *init_list() {
    node *head = (node *) malloc(sizeof(node));
    head->data = FS;
    head->fib_data = 0;

    node *cur = head;
    for (int i = 0; i < N; ++i) {
        node *temp = (node *) malloc(sizeof(node));
        temp->data = FS + i + 1;
        temp->fib_data = 0;
        cur->next = temp;
        cur = temp;
    }
    cur->next = NULL;
    return head;
}

int main() {
    printf("Process linked list\n");
    printf("  Each linked list node will be processed by function 'process_work()'\n");
    printf("  Each ll node will compute %d fibonacci numbers beginning with %d\n", N+1, FS);

    node *head = init_list();

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            node *curr = head;
            while (curr != NULL) {
                #pragma omp task firstprivate(curr)
                process_work(curr);
                curr = curr->next;
            }
        }
        #pragma omp taskwait
    }

    double end = omp_get_wtime();

    while (head != NULL) {
        printf("%d : %d\n", head->data, head->fib_data);
        node *tmp = head->next;
        free(head);
        head = tmp;
    }

    printf("Compute Time: %f seconds\n", end - start);
}