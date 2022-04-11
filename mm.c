/*
 * Simple, 64-bit allocator based on implicit free lists,
 * first fit placement, and boundary tag coalescing, as described
 * in the CS:APP2e text. Blocks must be aligned to 16 byte
 * boundaries. Minimum block size is 16 bytes.
 *
 * This version is loosely based on
 * http://csapp.cs.cmu.edu/3e/ics3/code/vm/malloc/mm.c
 * but unlike the book's version, it does not use C preprocessor
 * macros or explicit bit operations.
 *
 * It follows the book in counting in units of 4-byte words,
 * but note that this is a choice (my actual solution chooses
 * to count everything in bytes instead.)
 *
 * You may use this code as a starting point for your implementation
 * if you want.
 *
 * Adapted for CS3214 Summer 2020 by gback
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

struct boundary_tag
{
    int inuse : 1; // inuse bit
    int size : 31; // size of block, in words
                   // block size
};

/* FENCE is used for heap prologue/epilogue. */
const struct boundary_tag FENCE = {
    .inuse = 1,
    .size = 0};

/* A C struct describing the beginning of each block.
 * For implicit lists, used and free blocks have the same
 * structure, so one struct will suffice for this example.
 *
 * If each block is aligned at 12 mod 16, each payload will
 * be aligned at 0 mod 16.
 */
struct block
{
    struct boundary_tag header; /* offset 0, at address 12 mod 16 */
    char payload[0];            /* offset 4, at address 0 mod 16 */
};

struct free_blk
{
    struct boundary_tag header; /* offset 0, at address 12 mod 16 */
    struct list_elem elem;      /* position the block in the segregated list */
};

/* Basic constants and macros */
#define WSIZE sizeof(struct boundary_tag) /* Word and header/footer size (bytes) */
#define DSIZE 2 * sizeof(struct boundary_tag)
#define MIN_BLOCK_SIZE_WORDS 4 /* Minimum block size in words */
#define CHUNKSIZE (1 << 10)    /* Extend heap by this amount (words) */
#define NUM_LIST 20            /* Number of segregated list */

static inline size_t max(size_t x, size_t y)
{
    return x > y ? x : y;
}

static inline size_t min(size_t x, size_t y)
{
    return x < y ? x : y;
}

static size_t align(size_t size)
{
    return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

static bool is_aligned(size_t size) __attribute__((__unused__));

static bool is_aligned(size_t size)
{
    return size % ALIGNMENT == 0;
}

/* Global variables */
static struct list free_list[NUM_LIST];

/* Function prototypes for internal helper routines */
static struct free_blk *extend_heap(size_t words);
static void place(struct block *bp, size_t asize);
static struct free_blk *find_fit(size_t asize);
static struct free_blk *coalesce(struct free_blk *bp);

static void init_list();

/* Given a block, obtain previous's block footer.
   Works for left-most block also. */
static struct boundary_tag *prev_blk_footer(struct free_blk *blk)
{
    return &blk->header - 1;
}

/* Given a block, obtain next's block header.
   Works for left-most block also. */
static struct boundary_tag *next_blk_header(struct free_blk *blk)
{
    return (struct boundary_tag *)((size_t *)blk + blk->header.size);
}

/* Return if block is free */
static bool blk_free(struct block *blk)
{
    return !blk->header.inuse;
}

/* Return size of block is free */
static size_t blk_size(struct free_blk *blk)
{
    return blk->header.size;
}

/* Given a list element, return the block. */
static struct free_blk *get_block(struct list_elem *e)
{
    return (struct free_block *)((size_t *)e - sizeof(struct boundary_tag) / WSIZE);
}

/* Given a block, obtain pointer to previous block.
   Not meaningful for left-most block. */
static struct free_blk *prev_blk(struct free_blk *blk)
{
    struct boundary_tag *prevfooter = prev_blk_footer(blk);
    assert(prevfooter->size != 0);
    return (struct free_blk *)((size_t *)blk - prevfooter->size);
}

/* Given a block, obtain pointer to next block.
   Not meaningful for right-most block. */
static struct free_blk *next_blk(struct free_blk *blk)
{
    assert(blk_size(blk) != 0);
    return (struct free_blk *)((size_t *)blk + blk->header.size);
}

/* Given a block, obtain its footer boundary tag */
static struct boundary_tag *get_footer(struct block *blk)
{
    return (void *)((size_t *)((size_t *)blk + ((struct free_blk *)blk)->header.size) - sizeof(struct boundary_tag) / WSIZE);
}

/* Set a block's size and inuse bit in header and footer */
static void set_header_and_footer(struct block *blk, int size, int inuse)
{
    blk->header.inuse = inuse;
    blk->header.size = size;
    *get_footer(blk) = blk->header; /* Copy header to footer */
}

/* Check if the boundary_tag is FENCE */
static bool is_fence(void *bt)
{
    return ((struct boundary_tag *)bt)->size == 0 && ((struct boundary_tag *)bt)->inuse == 1;
}

/* Mark a block as used and set its size. */
static void mark_block_used(struct block *blk, int size)
{
    set_header_and_footer(blk, size, 1);
}

/* Mark a block as free and set its size. */
static void mark_block_free(struct free_blk *blk, int size)
{
    set_header_and_footer(blk, size, 0);
}

/*
 * mm_init - Initialize the memory manager
 */
int mm_init(void)
{
    init_list();
    // assert (offsetof(struct block, payload) == 4);
    // assert (sizeof(struct boundary_tag) == 4);

    /* Create the initial empty heap */
    struct boundary_tag *initial = mem_sbrk(2 * sizeof(struct boundary_tag));
    if (initial == (void *)-1)
        return -1;

    /* We use a slightly different strategy than suggested in the book.
     * Rather than placing a min-sized prologue block at the beginning
     * of the heap, we simply place two fences.
     * The consequence is that coalesce() must call prev_blk_footer()
     * and not prev_blk() because prev_blk() cannot be called on the
     * left-most block.
     */
    initial[0] = FENCE; /* Prologue footer */
    // heap_list = (struct block *)&initial[3];
    initial[1] = FENCE; /* Epilogue header */

    /* Extend the empty heap with a free block of CHUNKSIZE bytes */
    if (extend_heap(CHUNKSIZE) == NULL)
        return -1;
    return 0;
}

/* Initialize a free list */
static void init_list()
{
    for (int i = 0; i < NUM_LIST; i++)
    {
        list_init(&free_list[i]);
    }
}

/*
 * mm_malloc - Allocate a block with at least size bytes of payload
 */
void *mm_malloc(size_t size)
{
    struct block *bp;
    struct block *new_block; /*a new block object*/

    /* If size less than 512 then round up*/
    if (size < 512)
    {
        int i = 0;
        int t_size = 1;

        while ((i < NUM_LIST - 1) && (t_size < size))
        {
            t_size <<= 1;
            i++;
        }
        size = t_size;
    }

    if (free_list[0].head.next == NULL)
    {
        mm_init();
    }

    /* Ignore spurious requests */
    if (size == 0)
        return NULL;

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
    if ((bp = extend_heap(extendwords)) == NULL)
        return NULL;

    new_block = bp;
    bp = place(new_block, awords);
    return bp->payload;
}

/*
 * mm_free - Free a block
 */
void mm_free(void *bp)
{
    // assert (heap_listp != 0);       // assert that mm_init was called
    if (bp == 0)
        return;

    /* Find block from user pointer */
    struct free_blk *free_blk = bp - offsetof(struct block, payload);

    if (free_list[0].head.next == NULL)
        mm_init();

    mark_block_free(free_blk, blk_size(free_blk));
    coalesce(free_blk);
}

/*
 * coalesce - Boundary tag coalescing. Return ptr to coalesced block
 */
static struct free_blk *coalesce(struct free_blk *bp)
{
    bool prev_alloc = prev_blk_footer(bp)->inuse; /* is previous block allocated? */
    bool next_alloc = next_blk_header(bp)->inuse; /* is next block allocated? */
    size_t size = blk_size(bp);

    if (prev_alloc && next_alloc)
    { /* Case 1 */
        // both are allocated, nothing to coalesce
        // return bp;
        insert(bp, size);
    }

    else if (prev_alloc && !next_alloc)
    { /* Case 2 */
        // combine this block and next block by extending it
        list_remove(&next_blk(bp)->elem);
        mark_block_free(bp, size + blk_size(next_blk(bp)));
        insert(bp, blk_size(bp));
    }

    else if (!prev_alloc && next_alloc)
    { /* Case 3 */
        // combine previous and this block by extending previous
        bp = prev_blk(bp);
        list_remove(&bp->elem);
        mark_block_free(bp, size + blk_size(bp));
        insert(bp, blk_size(bp));
    }

    else
    { /* Case 4 */
        // combine all previous, this, and next block into one
        list_remove(&next_blk(bp)->elem);
        list_remove(&prev_blk(bp)->elem);
        mark_block_free(prev_blk(bp), size + blk_size(next_blk(bp)) + blk_size(prev_blk(bp)));
        bp = prev_blk(bp);
        insert(bp, blk_size(bp));
    }
    return bp;
}

/*
 * mm_realloc - Naive implementation of realloc
 */
void *mm_realloc(void *ptr, size_t size)
{

    /* If size == 0 then this is just free, and we return NULL. */
    if (size == 0)
    {
        mm_free(ptr);
        return 0;
    }

    /* If oldptr is NULL, then this is just malloc. */
    if (ptr == NULL)
    {
        return mm_malloc(size);
    }

    void *newptr = mm_malloc(size);

    /* If realloc() fails the original block is left untouched  */
    if (!newptr)
    {
        return 0;
    }

    /* Copy the old data. */
    struct block *oldblock = ptr - offsetof(struct block, payload);
    size_t oldsize = blk_size(oldblock) * WSIZE;

    /* Adjust block size to include overhead and alignment reqs. */
    size += 2 * sizeof(struct boundary_tag); /* account for tags */

    /* Adjusted block size in words */
    size_t awords = max(MIN_BLOCK_SIZE_WORDS, align(size) / WSIZE); /* respect minimum size */

    /*This is the next block pointer*/
    struct free_blk *new_bp = (struct free_blk *)((size_t *)oldblock + oldsize);

    /*Case 1: When the break pointer is at the last block in the heap*/
    if (is_fence(new_bp))
    {
        size_t extendwords = max(awords - oldsize, CHUNKSIZE);
        if ((new_bp = (void *)extend_heap(extendwords)) == NULL)
        {
            return NULL;
        }
        list_remove(&new_bp->elem);
        mark_block_used(oldblock, oldsize + blk_size(new_bp));

        return ptr;
    }

    /*Case 2: When the next block pointer is not used.*/
    if (new_bp->header.inuse == 0)
    {   
        /*If the next block pointer is free and there is space for reallocation.*/
        if (awords <= oldsize + blk_size(new_bp))
        {
            size_t new_size = blk_size(new_bp);
            if (oldsize + new_size - awords >= MIN_BLOCK_SIZE_WORDS)
            {
                list_remove(&new_bp->elem);
                mark_block_used(oldblock, awords);
                struct free_blk *new_blk = (struct free_blk *)((size_t *)oldblock + oldblock->header.size);
                mark_block_free(new_blk, oldsize + new_size - awords);
                insert(new_blk, blk_size(new_blk));
            }
            else
            {
                list_remove(&new_bp->elem);
                mark_block_used(oldblock, oldsize + new_size);
            }

            return ptr;
        }
        /*If the next block pointer is free and there is no space for reallocation.*/
        else
        {
            if (is_fence(new_blk(new_bp)))
            {
                size_t extendwords = max(awords - oldsize - blk_size(new_bp), CHUNKSIZE);
                if ((void *)extend_heap(extendwords) == NULL)
                {
                    return NULL;
                }
                list_remove(&new_bp->elem);
                mark_block_used(oldblock, oldsize + blk_size(new_bp));

                return ptr;
            }
        }
    }

    /*Case 3: When the new size is less than the oldsize*/
    if (awords <= oldsize)
    {
        if (oldsize - awords >= MIN_BLOCK_SIZE_WORDS)
        {
            mark_block_used(oldblock, awords);
            struct free_blk *new_bp = (struct free_blk *)((size_t *)oldblock + awords);
            mark_block_free(new_bp, oldsize - awords);
            insert(new_bp, blk_size(new_bp));
        }
        return ptr;
    }

    /*Copy the old data.*/
    oldsize *= WSIZE;
    memcpy(newptr, ptr, oldsize);

    /* Free the old block. */
    mm_free(ptr);

    return newptr;
}

/*
 * checkheap - We don't check anything right now.
 */
void mm_checkheap(int verbose)
{
}

/*
 * The remaining routines are internal helper routines
 */

/*
 * extend_heap - Extend heap with free block and return its block pointer
 */
static struct free *extend_heap(size_t words)
{
    void *bp = mem_sbrk(words * WSIZE);

    if ((intptr_t)bp == -1)
        return NULL;

    /* Initialize free block header/footer and the epilogue header.
     * Note that we overwrite the previous epilogue here. */
    struct block *blk = bp - sizeof(FENCE);
    mark_block_free(blk, words);
    next_blk(blk)->header = FENCE;

    /* Coalesce if the previous block was free */
    return coalesce(blk);
}

/*
 * place - Place block of asize words at start of free block bp
 *         and split if remainder would be at least minimum block size
 */
static void place(struct block *bp, size_t asize)
{
    size_t csize = blk_size(bp);

    if ((csize - asize) >= MIN_BLOCK_SIZE_WORDS)
    {
        mark_block_used(bp, asize);
        bp = next_blk(bp);
        mark_block_free(bp, csize - asize);
    }
    else
    {
        mark_block_used(bp, csize);
    }
}

/*
 * find_fit - Find a fit for a block with asize words
 */
static struct free_blk *find_fit(size_t asize)
{
    /* First fit search */
    for (struct free_blk *bp = heap_list; blk_size(bp) > 0; bp = next_blk(bp))
    {
        if (blk_free(bp) && asize <= blk_size(bp))
        {
            return bp;
        }
    }
    return NULL; /* No fit */
}

team_t team = {
    /* Team name */
    "Sample allocator using implicit lists",
    /* First member's full name */
    "Godmar Back",
    "gback@cs.vt.edu",
    /* Second member's full name (leave blank if none) */
    "",
    "",
};
