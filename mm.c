/*
 * Simple, 64-bit allocator based on explicit free lists,
 * first fit placement, and boundary tag coalescing, as described
 * in the CS:APP2e text. Blocks must be aligned to 16 byte
 * boundaries. Minimum block size is 16 bytes.
 *
 * This version is loosely based on
 * http://csapp.cs.cmu.edu/3e/ics3/code/vm/malloc/mm.c
 * but unlike the book's version, it does not use C preprocessor
 * macros or explicit bit operations.
 *
 * Our implementation is adapted from the provided code by Dr. Back
 */

#include "mm.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "list.h"
#include "memlib.h"

/* The boundary tag structure */
struct boundary_tag {
    size_t inuse : 1;  // inuse bit
    size_t size : 31;  // size of block, in words
                       // block size
};

/* FENCE is used for heap prologue/epilogue. */
const struct boundary_tag FENCE = {
    .inuse = 1,  // inuse bit
    .size = 0    // size of block, in words
};

/* A C struct describing the beginning of each block.
 * For implicit lists, used and free blocks have the same
 * structure, so one struct will suffice for this example.
 *
 * If each block is aligned at 12 mod 16, each payload will
 * be aligned at 0 mod 16.
 */
struct alloc_blk {
    struct boundary_tag header; /* offset 0, at address 12 mod 16 */
    char payload[0];            /* offset 4, at address 0 mod 16 */
};

/*
 * A free block struct without payload, which contains a header and
 * an element struct such that it can be placed and located in the list.
 *
 * If each block is aligned at 12 mod 16, each payload will
 * be aligned at 0 mod 16.
 */
struct free_blk {
    struct boundary_tag header; /* offset 0, at address 12 mod 16 */
    struct list_elem elem;      /* position the block in the segregated list */
};

/* Basic constants and macros */
#define WSIZE sizeof(struct boundary_tag)     /* Word and header/footer size (bytes) */
#define DSIZE 2 * sizeof(struct boundary_tag) /* Doubleword size (bytes) */
#define MIN_BLOCK_SIZE_WORDS 4                /* Minimum block size in words */
#define CHUNKSIZE (1 << 10)                   /* Extend heap by this amount (words) */
#define NUM_LIST 10                           /* Number of segregated list */

/* Return the largest number */
static inline size_t max(size_t x, size_t y) {
    return x > y ? x : y;
}

/* Algin to double word size */
static size_t align(size_t size) {
    return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

/* Check if the block is aligned */
static bool is_aligned(size_t size) __attribute__((__unused__));

/* Check if the block is aligned */
static bool is_aligned(size_t size) {
    return size % ALIGNMENT == 0;
}

/* Global variables */
static struct list segregated_list[NUM_LIST];   /* free block list */
static struct alloc_blk *heap_listp = 0;        /* Pointer to first block */

/* Function prototypes for internal helper routines */
static struct free_blk *extend_heap(size_t words);
static void *place(void *bp, size_t asize);
static void *find_fit(size_t asize);
static struct free_blk *coalesce(struct free_blk *bp);
static void push_free_blk(struct free_blk *bp, size_t size);
static void init_list();
static int find_position(size_t size);
static int round_up(size_t size);
static void split_blk(struct alloc_blk *oldblock, size_t ori_size, size_t req_size);

/* Given a block, obtain previous's block footer.
   Works for left-most block also. */
static struct boundary_tag *prev_blk_footer(struct free_blk *blk) {
    return &blk->header - 1;
}

/* Return if block is free */
static bool blk_free(struct free_blk *blk) {
    return !blk->header.inuse;
}

/* Return the size of block */
static size_t blk_size(struct free_blk *blk) {
    return blk->header.size;
}

/* Given a block, obtain pointer to previous block.
   Not meaningful for left-most block. */
static struct free_blk *prev_blk(struct free_blk *blk) {
    struct boundary_tag *prevfooter = prev_blk_footer(blk);
    assert(prevfooter->size != 0);
    return (struct free_blk *)((void *)blk - WSIZE * prevfooter->size);
}

/* Given a block, obtain pointer to next block.
   Not meaningful for right-most block. */
static struct free_blk *next_blk(struct free_blk *blk) {
    assert(blk_size(blk) != 0);
    return (struct free_blk *)((void *)blk + WSIZE * blk->header.size);
}

/* Given an allocated block, obtain its footer boundary tag */
static struct boundary_tag *get_footer(struct alloc_blk *blk) {
    return ((void *)blk + WSIZE * blk->header.size) - sizeof(struct boundary_tag);
}

/* Given a free block, obtain its footer boundary tag */
static struct boundary_tag *get_footer_free(struct free_blk *blk) {
    return ((void *)blk + WSIZE * blk->header.size) - sizeof(struct boundary_tag);
}

/* Set an allocated block's size and inuse bit in header and footer */
static void set_header_and_footer(struct alloc_blk *blk, int size, int inuse) {
    blk->header.inuse = inuse;
    blk->header.size = size;
    *get_footer(blk) = blk->header; /* Copy header to footer */
}

/*Set a free block'size and inuse bit in header and footer for free blocks.*/
static void set_header_and_footer_free(struct free_blk *blk, int size, int inuse) {
    blk->header.inuse = inuse;
    blk->header.size = size;
    *get_footer_free(blk) = blk->header; /* Copy header to footer */
}

/* Mark an allocated block as used and set its size. */
static void mark_block_used(struct alloc_blk *blk, int size) {
    set_header_and_footer(blk, size, 1);
}

/* Mark a free block as free and set its size. */
static void mark_block_free(struct free_blk *blk, int size) {
    set_header_and_footer_free(blk, size, 0);
}

/*
 * mm_init - Initialize the memory manager
 *
 * Allocate the initial heap area, and place two fences for the prologue header
 * and epilogue footer. As well as, initialize the segregated list.
 *
 * Return -1 if there was a problem in initializing the heap. Return 0 otherwise.
 */
int mm_init(void) {
    init_list();
    assert(offsetof(struct alloc_blk, payload) == 4);
    assert(sizeof(struct boundary_tag) == 4);
    assert(sizeof(struct free_blk) == 4);

    /* Create the initial empty heap */
    struct boundary_tag *initial = mem_sbrk(4 * sizeof(struct boundary_tag));
    if (initial == (void *)-1)
        return -1;

    /* We use a slightly different strategy than suggested in the book.
     * Rather than placing a min-sized prologue block at the beginning
     * of the heap, we simply place two fences.
     * The consequence is that coalesce() must call prev_blk_footer()
     * and not prev_blk() because prev_blk() cannot be called on the
     * left-most block.
     */
    initial[2] = FENCE; /* Prologue footer */
    heap_listp = (struct alloc_blk *)&initial[3];
    initial[3] = FENCE; /* Epilogue header */

    /* Extend the empty heap with a free block of CHUNKSIZE bytes */
    if (extend_heap(CHUNKSIZE) == NULL) {
        return -1;
    }
    return 0;
}

/*
 * mm_malloc - Allocate a block with at least size bytes of payload
 *
 * Adjust the size to fit alignment requirements, and then find a fit
 * for the block with adjusted size. If no fit is found, extend the heap
 * to place the block.
 *
 * Return a pointer to an allocated block payload of at least size bytes.
 */
void *mm_malloc(size_t size) {
    struct alloc_blk *bp;        /*a new alloc_blk object*/

    /* Ignore spurious requests */
    if (size == 0) {
        return NULL;
    }

    /* If size less than 512 then round up */
    if (size < 512) size = round_up(size);

    /* If heap is not allocated, call mm_init() */
    if (heap_listp == 0) {
        mm_init();
    }

    /* Adjust block size to include overhead and alignment reqs */
    size += 2 * sizeof(struct boundary_tag); /* account for tags */

    /* Adjusted block size in words */
    size_t req_size = max(MIN_BLOCK_SIZE_WORDS, align(size) / WSIZE); /* respect minimum size */

    /* Search the free list for a fit */
    if ((bp = find_fit(req_size)) != NULL) {
        bp = place(bp, req_size);
        return bp->payload;
    }

    /* No fit found. Get more memory and place the block */
    size_t extendwords = max(req_size, CHUNKSIZE); /* Amount to extend heap if no fit */
    if ((bp = (struct alloc_blk *)extend_heap(extendwords)) == NULL)
        return NULL;

    bp = place(bp, req_size);

    /* Returns a pointer to an allocated block payload.*/
    return bp->payload;
}

/*
 * mm_free - Free a block pointed to by bp. This function work whn the passed in pointer
 * was returuend by the call to mm_malloc and mm_realloc and has not been freed yet.
 *
 * The function returns nothing
 */
void mm_free(void *bp) {
    assert(segregated_list != NULL);  // assert that mm_init was called
    assert(heap_listp != 0);          // assert that mm_init was called

    /*Return nothing*/
    if (bp == 0)
        return;

    /* Find block from user pointer */
    struct free_blk *free_blk = bp - offsetof(struct alloc_blk, payload);

    /* Set the block as free */
    mark_block_free(free_blk, blk_size(free_blk));

    /* Coalesce the block */
    coalesce(free_blk);
}

/*
 * mm_realloc - Implementation of realloc
 *
 * If the ptr is NULL, then call mm_malloc with the given size.
 * If the size is 0, then call mm_free and return NULL.
 * If the ptr is not NULL, meaning that it has returned from a previous call to
 * mm_malloc, or mm_realloc. Then mm_realloc should change the block size to the
 * given value, and return the address of the new block. The contents of the new
 * block should be the same as the old block.
 *
 *  Returns a pointer to an allocated region of at least size bytes
 */
void *mm_realloc(void *ptr, size_t size) {
    /* If size == 0 then this is just free, and we return NULL. */
    if (size == 0) {
        mm_free(ptr);
        return 0;
    }

    /* If ptr is NULL, then this is just malloc. */
    if (ptr == NULL) {
        return mm_malloc(size);
    }

    size_t temp_size = size;

    /* Copy the old data. */
    struct alloc_blk *oldblock = ptr - offsetof(struct alloc_blk, payload);
    size_t ori_size = oldblock->header.size;

    /* Adjust block size to include overhead and alignment reqs. */
    size += 2 * sizeof(struct boundary_tag); /* account for tags */

    /* Adjusted block size in words */
    size_t req_size = max(MIN_BLOCK_SIZE_WORDS, align(size) / WSIZE); /* respect minimum size */

    /* Case 0: The requested size is smaller than the original size */
    if (req_size <= ori_size) {
        if (ori_size - req_size >= MIN_BLOCK_SIZE_WORDS) {
            split_blk(oldblock, ori_size, req_size);
        }
        return ptr;
    }

    /*This is the next block pointer*/
    struct free_blk *next_bp = next_blk((struct free_blk *)oldblock);
    bool next_alloc = !blk_free(next_bp);    /* is next block allocated? */

    /* If the next block is not allocated */
    if (!next_alloc) {
        /* Case 1: The requested size is smaller than the combined block size */
        if (req_size <= ori_size + blk_size(next_bp)) {
            size_t temp_size = blk_size(next_bp);
            if (ori_size + temp_size - req_size >= MIN_BLOCK_SIZE_WORDS) {
                list_remove(&next_bp->elem);
                split_blk(oldblock, ori_size + temp_size, req_size);
            } else {
                list_remove(&next_bp->elem);
                mark_block_used(oldblock, ori_size + temp_size);
            }
            return ptr;
        }
    }

    /*Case 2: The block pointer is at the last block in the heap */
    if (next_bp->header.size == 0 && next_bp->header.inuse == 1) {
        size_t extendwords = max(req_size - ori_size, CHUNKSIZE);
        if ((next_bp = extend_heap(extendwords)) == NULL) {
            return NULL;
        }
        list_remove(&next_bp->elem);
        mark_block_used(oldblock, ori_size + blk_size(next_bp));
        return ptr;
    }

    void *newptr = mm_malloc(temp_size);

    /* If realloc() fails the original block is left untouched  */
    if (!newptr) {
        return 0;
    }

    /*Copy the old data.*/
    ori_size *= WSIZE;
    memmove(newptr, ptr, ori_size);

    /* Free the old block. */
    mm_free(ptr);
    return newptr;
}

/* ========== The remaining routines are internal helper routines ========== */

/*
 * Initialize 10 segregated list with provided list.h functions
 */
static void init_list() {
    for (int i = 0; i < NUM_LIST; i++) {
        list_init(&segregated_list[i]);
    }
}

/*
 * coalesce - Boundary tag coalescing. Return ptr to coalesced block
 *
 * This function takes in a pointer to a free block and check its
 * neightboring blocks to see if they are free. If both neighbors are allocated,
 * then push the current block to the segregated list. If one of the neighbors is
 * free, then merge them and push the merged block to the segregated list. If all
 * neighbors are free, then push the merged three block to the segregated list.
 *
 * Return a pointer to the coalesced block.
 */
static struct free_blk *coalesce(struct free_blk *bp) {
    bool prev_alloc = prev_blk_footer(bp)->inuse; /* is previous block allocated? */
    bool next_alloc = !blk_free(next_blk(bp));    /* is next block allocated? */
    size_t size = blk_size(bp);

    if (prev_alloc && next_alloc) { /* Case 1 */
        // both are allocated, nothing to coalesce
        push_free_blk(bp, size);
    }

    else if (prev_alloc && !next_alloc) { /* Case 2 */
        // combine this block and next block by extending it
        list_remove(&next_blk(bp)->elem);
        mark_block_free(bp, size + blk_size(next_blk(bp)));
        push_free_blk(bp, blk_size(bp));
    }

    else if (!prev_alloc && next_alloc) { /* Case 3 */
        // combine previous and this block by extending previous
        bp = prev_blk(bp);
        list_remove(&bp->elem);
        mark_block_free(bp, size + blk_size(bp));
        push_free_blk(bp, blk_size(bp));
    }

    else { /* Case 4 */
        // combine all previous, this, and next block into one
        list_remove(&next_blk(bp)->elem);
        list_remove(&prev_blk(bp)->elem);
        mark_block_free(prev_blk(bp), size + blk_size(next_blk(bp)) + blk_size(prev_blk(bp)));
        bp = prev_blk(bp);
        push_free_blk(bp, blk_size(bp));
    }

    return bp;
}

/*
 * push_free_blk - Push a free block onto the free list
 *
 * It will find the coresponding segregated list and push the block to the list.
 */
static void push_free_blk(struct free_blk *bp, size_t size) {
    struct list_elem *before = &bp->elem;
    // choose the list with appropriate size
    int i = 0;
    while (i < NUM_LIST - 1 && size > 1) {
        size = size >> 1;
        i++;
    }

    //printf("push_free_blk: size: %ld, i: %d\n", size, i);

    if (list_empty(&segregated_list[i])) {
        list_push_front(&segregated_list[i], before);
    } else {
        struct list_elem *e = list_begin(&segregated_list[i]);
        // struct free_blk *blk = list_entry(e, struct free_blk, elem);
        // size_t free_size = blk_size(blk);
        while (e != list_end(&segregated_list[i])) {
            struct free_blk *blk = list_entry(e, struct free_blk, elem);
            size_t free_size = blk_size(blk);
            if (size <= free_size) {
                break;
            }
            e = list_next(e);
        }
        list_insert(e, before);
    }
}

/*
 * extend_heap - Extend heap with free block
 *
 * Before extending the heap, it makes sure that the size of the word is rounded to
 * the ALIGNMENT, then it checks if it is larger than the MIN_BLOCK_SIZE_WORDS. Finally,
 * it will extend the heap and mark the new block as free.
 *
 * Return a pointer to the new free block.
 */
static struct free_blk *extend_heap(size_t words) {
    void *bp = mem_sbrk(words * WSIZE);

    if ((intptr_t)bp == -1)
        return NULL;

    /* Initialize free block header/footer and the epilogue header.
     * Note that we overwrite the previous epilogue here. */
    struct free_blk *blk = bp - sizeof(FENCE);
    mark_block_free(blk, words);
    next_blk(blk)->header = FENCE;

    /* Coalesce if the previous block was free */
    return coalesce(blk);
}

/*
 * place - Place block of asize words at start of free block pointer
 *
 * Compare the size of the free block to the size of the block to be allocated.
 * If the free block is larger than the block to be allocated and their difference
 * is larger than the minimum block size, then split the free block into two blocks.
 * Otherwise, just place the block to the free block.
 *
 */
static void *place(void *bp, size_t asize) {
    size_t csize = blk_size(bp);

    if ((csize - asize) >= MIN_BLOCK_SIZE_WORDS) {
        mark_block_free((struct free_blk *)bp, csize - asize);
        void *temp = next_blk(bp);
        mark_block_used((struct alloc_blk *)temp, asize);
        return temp;
    } else {
        list_remove(&((struct free_blk *)bp)->elem);
        mark_block_used(bp, csize);
        return bp;
    }
}

/*
 * find_fit - Find a fit for a block with asize words
 *
 * Find the segregated list with the appropriate size and check if
 * the size of the free block is large enough. If so, return the free block.
 * Otherwise, return NULL to indicate there is no fit.
 */
static void *find_fit(size_t asize) {
    /* Find the corresponding list location */
    int list_count = find_position(asize);
    struct free_blk *blk_ptr;

    /* If traverse a list more than 20 times, then jump to the next list */
    int threshold = 20;  
    /*Continue if the list is empty*/
    for (; list_count < NUM_LIST; list_count++) {
        if (list_empty(&segregated_list[list_count])) {
            continue;
        }
        int count = 0;
        /*Check the element in the list*/
        for (struct list_elem *element = list_begin(&segregated_list[list_count]); 
            element != list_end(&segregated_list[list_count]); 
            element = list_next(element)) {

            if (count > threshold) {
                break;
            }
            
            blk_ptr = list_entry(element, struct free_blk, elem);
            /* Return the block pointer if the size of block pointer is larger than asize words*/
            if (blk_size(blk_ptr) >= asize) {
                return blk_ptr;
            }
            count++;
        }
    }
    return NULL; /* No fit */
}

/*
 * find_position - Find the position of the segregated list
 *
 * It will find the coresponding segregated list and return the position.
 * This will be the index of the segregated list to start traversing.
 */
static int find_position(size_t asize) {
    int list_count = 0;
    size_t temp_size = asize;
    while ((list_count < NUM_LIST - 1) && (temp_size > 1)) {
        temp_size = temp_size >> 1;
        list_count++;
    }
    return list_count;
}

/*
 * round_up - Round up to the 2^n using bitwise operation
 * 
 * Return the rounded-up value.
 */
static int round_up(size_t size) {
    int i = 0;
    int t_size = 1;

    while ((i < NUM_LIST - 1) && (t_size < size)) {
        t_size = t_size << 1;
        i++;
    }
    return t_size;
}

/* split_block - Split an allocated block into two blocks
 *
 * It will split the block if there is enough space for a 
 * MIN_BLOCK_SIZE_WORDS.
 *
 * The function does not return anything.
 */
static void split_blk(struct alloc_blk *oldblock, size_t ori_size, size_t req_size) {
    mark_block_used(oldblock, req_size);
    struct free_blk *bp = next_blk((struct free_blk *)oldblock);
    mark_block_free(bp, ori_size - req_size);
    push_free_blk(bp, blk_size(bp));
}

/* Team info */
team_t team = {
    /* Team name */
    "Kevin and Kevin",
    /* First member's full name */
    "Mingkai Pang",
    "pangmin@vt.edu",
    /* Second member's full name (leave blank if none) */
    "Jiayue Lin",
    "jiayuelin@vt.edu",
};
