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

struct free_blk {
    struct boundary_tag header; /* offset 0, at address 12 mod 16 */
    struct list_elem elem;      /* position the block in the segregated list */
};

/* Basic constants and macros */
#define WSIZE sizeof(struct boundary_tag)     /* Word and header/footer size (bytes) */
#define DSIZE 2 * sizeof(struct boundary_tag) /* Doubleword size (bytes) */
#define MIN_BLOCK_SIZE_WORDS 4                /* Minimum block size in words */
#define CHUNKSIZE (1 << 10)                   /* Extend heap by this amount (words) */
#define NUM_LIST 20                           /* Number of segregated list */

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
static struct list segregated_list[NUM_LIST];

/* Function prototypes for internal helper routines */
static struct free_blk *extend_heap(size_t words);
static void *place(void *bp, size_t asize);
static void *find_fit(size_t asize);
static struct free_blk *coalesce(struct free_blk *bp);
static void push_free_blk(struct free_blk *bp, size_t size);
static void init_list();

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

/* Given a list element, return the block. */
static struct free_blk *get_block(struct list_elem *e) {
    return (struct free_blk *)((size_t *)e - sizeof(struct boundary_tag) / WSIZE);
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

/* Check if the boundary_tag reaches a FENCE */
static bool is_fence(void *bt) {
    return ((struct boundary_tag *)bt)->size == 0 && ((struct boundary_tag *)bt)->inuse == 1;
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
    initial[3] = FENCE; /* Epilogue header */

    /* Extend the empty heap with a free block of CHUNKSIZE bytes */
    if (extend_heap(CHUNKSIZE) == NULL)
        return -1;
    return 0;
}

/*
 * mm_malloc - Allocate a block with at least size bytes of payload
 */
void *mm_malloc(size_t size)
{
    struct alloc_blk *bp; /*a new alloc_blk object*/
    struct alloc_blk *new_block; /*a new alloc_blk object*/

    /* Ignore spurious requests */
    if (size == 0)
    {
        return NULL;
    }

    /* If size less than 512 then round up */
    if (size < 512)
    {
        int i = 0;
        int t_size = 1;

        while ((i < NUM_LIST - 1) && (t_size < size))
        {
            t_size = t_size << 1;
            i++;
        }
        size = t_size;
    }

    /* If the segregated list is empty, then call mm_init() for initialization*/
    if (segregated_list[0].head.next == NULL)
    {
        mm_init();
    }

    /* Adjust block size to include overhead and alignment reqs. */
    size += 2 * sizeof(struct boundary_tag); /* account for tags */

    /* Adjusted block size in words */
    size_t awords = max(MIN_BLOCK_SIZE_WORDS, align(size) / WSIZE); /* respect minimum size */

    /* Search the free list for a fit */
    if ((bp = find_fit(awords)) != NULL)
    {
        new_block = bp;
        bp = place(new_block, awords);
        return bp->payload;
    }

    /* No fit found. Get more memory and place the block */
    size_t extendwords = max(awords, CHUNKSIZE); /* Amount to extend heap if no fit */
    if ((bp = (struct alloc_blk *)extend_heap(extendwords)) == NULL)
        return NULL;

    new_block = bp;
    bp = place(new_block, awords);
    
    /* Returns a pointer to an allocated block payload.*/
    return bp->payload;
}

/*
 * mm_free - Free a block
 */
void mm_free(void *bp)
{
    assert(segregated_list != NULL); // assert that mm_init was called
    
    /*Return nothing*/
    if (bp == 0)
        return;

    /* Find block from user pointer */
    struct free_blk *free_blk = bp - offsetof(struct alloc_blk, payload);

    /* If the segregated list is empty, then call mm_init() for initialization*/
    if (segregated_list[0].head.next == NULL)
        mm_init();

    /* Set the block as free */
    mark_block_free(free_blk, blk_size(free_blk));
    
    /* Coalesce the block */
    coalesce(free_blk);
}

/*
 * mm_realloc - Naive implementation of realloc
 */
void *mm_realloc(void *ptr, size_t size)
{
    size_t osize = size;

    /* If the pointer block is NULL, realloc should be the same as malloc */
    if (ptr == NULL)
        return mm_malloc(size);

    /* If the size if 0, free the block and return 0 */
    if (size == 0)
    {
        mm_free(ptr);
        return NULL;
    }

    size_t oldsize;
    size_t extendwords;

    struct alloc_blk *oldblock = ptr - offsetof(struct alloc_blk, payload);
    oldsize = oldblock->header.size;

    size += 2 * sizeof(struct boundary_tag);                /* account for tags */
    size = (size + DSIZE - 1) & ~(DSIZE - 1);               /* align to double word */
    size_t asize = max(MIN_BLOCK_SIZE_WORDS, size / WSIZE); /* respect minimum size */

    // In the following four cases we can eliminate copying the payload

    /* Case 1: new size is smaller than oldsize, split block and return ptr */
    if (asize <= oldsize)
    {
        if (oldsize - asize >= MIN_BLOCK_SIZE_WORDS)
        {
            mark_block_used(oldblock, asize);
            struct free_blk *next_bp = (struct free_blk *)((size_t *)oldblock + asize);
            mark_block_free(next_bp, oldsize - asize);
            push_free_blk(next_bp, blk_size(next_bp));
        }
        return ptr;
    }

    // Get next block
    struct free_blk *next_bp = (struct free_blk *)((size_t *)oldblock + oldsize);

    /* Case 2: ptr is the last block in the heap, extend the heap and coalesce mutually */
    if (is_fence(next_bp))
    {
        extendwords = max(asize - oldsize, CHUNKSIZE);
        if ((next_bp = (void *)extend_heap(extendwords)) == NULL)
            return NULL;
        list_remove(&next_bp->elem);
        mark_block_used(oldblock, oldsize + blk_size(next_bp));
        return ptr;
    }

    // If next block is free
    if (next_bp->header.inuse == 0)
    {
        /* Case 3: next block is a free block and have enough space to reallocate,
           coalesce two blocks mutually */
        if (asize <= oldsize + blk_size(next_bp))
        {
            size_t next_size = blk_size(next_bp);
            if (oldsize + next_size - asize >= MIN_BLOCK_SIZE_WORDS)
            {
                list_remove(&next_bp->elem);
                mark_block_used(oldblock, asize);
                struct free_blk *new_blk = (struct free_blk *)((size_t *)oldblock + oldblock->header.size);
                mark_block_free(new_blk, oldsize + next_size - asize);
                push_free_blk(new_blk, blk_size(new_blk));
            }
            else
            {
                list_remove(&next_bp->elem);
                mark_block_used(oldblock, oldsize + next_size);
            }
            return ptr;
        }

        /* Case 4: next block is a free block but do not have enough space to reallocate,
           but next block is the last block in the heap,
           extend the heap and coalesce two blocks mutually */
        else
        {
            if (is_fence(next_blk(next_bp)))
            {
                extendwords = max(asize - oldsize - blk_size(next_bp), CHUNKSIZE);
                if ((void *)extend_heap(extendwords) == NULL)
                    return NULL;
                list_remove(&next_bp->elem);
                mark_block_used(oldblock, oldsize + blk_size(next_bp));
                return ptr;
            }
        }
    }

    void *newptr = mm_malloc(osize);

    /* If malloc() fails, the original block is left untouched  */
    if (!newptr)
    {
        return 0;
    }

    /* Copy the old data. */
    oldsize *= WSIZE;
    memcpy(newptr, ptr, oldsize);

    /* Free the old block. */
    mm_free(ptr);
    return newptr;
}

/* -------- The remaining routines are internal helper routines -------- */

/* Initialize a free list */
static void init_list()
{
    for (int i = 0; i < NUM_LIST; i++)
    {
        list_init(&segregated_list[i]);
    }
}

/*
 * coalesce - Boundary tag coalescing. Return ptr to coalesced block
 */
static struct free_blk *coalesce(struct free_blk *bp)
{
    bool prev_alloc = prev_blk_footer(bp)->inuse; /* is previous block allocated? */
    bool next_alloc = !blk_free(next_blk(bp));    /* is next block allocated? */
    size_t size = blk_size(bp);

    if (prev_alloc && next_alloc)
    { /* Case 1 */
        // both are allocated, nothing to coalesce
        // return bp;
        push_free_blk(bp, size);
    }

    else if (prev_alloc && !next_alloc)
    { /* Case 2 */
        // combine this block and next block by extending it
        list_remove(&next_blk(bp)->elem);
        mark_block_free(bp, size + blk_size(next_blk(bp)));
        push_free_blk(bp, blk_size(bp));
    }

    else if (!prev_alloc && next_alloc)
    { /* Case 3 */
        // combine previous and this block by extending previous
        bp = prev_blk(bp);
        list_remove(&bp->elem);
        mark_block_free(bp, size + blk_size(bp));
        push_free_blk(bp, blk_size(bp));
    }

    else
    { /* Case 4 */
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
 */
static void push_free_blk(struct free_blk *bp, size_t size)
{
    struct list_elem *before = &bp->elem;
    // choose the list with appropriate size
    int i = 0;
    while (i < NUM_LIST - 1 && size > 1)
    {
        size = size >> 1;
        i++;
    }

    if (list_empty(&segregated_list[i]))
    {
        list_push_front(&segregated_list[i], before);
    }
    else
    {
        struct list_elem *e = list_begin(&segregated_list[i]);
        size_t free_size = blk_size(get_block(e));
        while (e != list_end(&segregated_list[i]))
        {
            if (size <= free_size)
            {
                break;
            }

            e = list_next(e);
            free_size = blk_size(get_block(e));
        }
        list_insert(e, before);
    }
}

/*
 * extend_heap - Extend heap with free block and return its block pointer !!!
 */
static struct free_blk *extend_heap(size_t words)
{
    words = (words + 1) & ~1;                 /* align to double word boundary */
    words = max(words, MIN_BLOCK_SIZE_WORDS); /* compare to minimum block size, and get the max */

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
 * place - Place block of asize words at start of free block bp
 *         and split if remainder would be at least minimum block size !!!
 */
static void *place(void *bp, size_t asize)
{
    void *block;

    size_t csize = blk_size(bp);

    if ((csize - asize) >= MIN_BLOCK_SIZE_WORDS)
    {
        mark_block_free((struct free_blk *)bp, csize - asize);
        block = next_blk(bp);
        mark_block_used((struct alloc_blk *)block, asize);
        return block;
    }
    else
    {
        list_remove(&((struct free_blk *)bp)->elem);
        mark_block_used(bp, csize);
        return bp;
    }
}

/*
 * find_fit - Find a fit for a block with asize words
 */
static void *find_fit(size_t asize)
{
    /* First fit search */
    int new_list = 0;
    size_t new_size = asize;
    struct free_blk *blk_ptr;

    while ((new_list < NUM_LIST - 1) && (new_size > 1))
    {
        new_size = new_size >> 1;
        new_list++;
    }

    for (; new_list < NUM_LIST; new_list++)
    {
        if (list_empty(&segregated_list[new_list]))
        {
            continue;
        }

        for (struct list_elem *element = list_begin(&segregated_list[new_list]); element != list_end(&segregated_list[new_list]); element = list_next(element))
        {
            blk_ptr = get_block(element);

            if (blk_size(blk_ptr) >= asize)
            {
                return blk_ptr;
            }
        }
    }
    return NULL; /* No fit */
}

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
