#pragma once

#include <nmmintrin.h>  // Include for _mm_popcnt_u64
#include "cuda_utils.cuh"
#include "test.h"
#include "utils.h"

#define CUDA_DEBUG_REFERENCE_DECK 0  // Add this define to control reference deck debug output

// Cards are represented as either 4x 16-bit unsigned integers
// or a single 64-bit unsigned integer. The bits are slightly different in each case
// but the idea is the same. Each rank and suit get a bit.
// Hands are represented the same way and are simply the sum of the cards in the hand.
// The sum captures all the information needed to compute the hand type.
// (up to 7 cards per hand is supported, but in this implementation we only use 5)
//
// We store a reference deck in constant memory. It contains all 52 cards in ascending order
// using the above representation. In some cases a copy may be stores per thread in shared
// memory. This produces significant performance improvements.
// A separate deck of uint8_t indices may be stored per thread in shared memory
// to allow for efficient shuffling

struct __align__(8) CudaCard {
  union {
    uint64_t v = 0x0ull;
    struct {
      uint16_t v3;
      uint16_t v2;
      uint16_t v1;
      uint16_t v0;
    };
  };
};

__constant__ CudaCard REFERENCE_DECK[52];

__host__ __device__ inline CudaCard getCudaCard(uint8_t card) {
  uint8_t  r = card >> 2;
  uint8_t  s = card & 0x3;
  CudaCard c;
  c.v0 = static_cast<uint16_t>(r < 5 ? (0x1u << (3 * r)) : 0x0);
  c.v1 = static_cast<uint16_t>(r > 4 && r < 8 ? (0x1u << (3 * (r - 5))) : 0x0);
  c.v2 = static_cast<uint16_t>(r > 7 ? (0x1u << (3 * (r - 8))) : 0x0);
  c.v3 = static_cast<uint16_t>(0x1u << (3 * s));
  return c;
}

__host__ __device__ inline CudaCard getCudaCard64(uint8_t card) {
  // Version with single 64 bit storage.
  // Each rank and suit get 3 bits.
  // Ranks start at 0 and end at 38
  // Suits start at 39 and end at 50
  uint8_t  r = card >> 2;
  uint8_t  s = card & 0x3;
  CudaCard c;
  c.v = static_cast<uint64_t>(static_cast<uint64_t>(0x1) << (3 * s));   // First pack in the suit
  c.v <<= 39;                                                           // Shift to the suit
  c.v |= static_cast<uint64_t>(static_cast<uint64_t>(0x1) << (3 * r));  // Add the rank

  return c;
}

void initializeReferenceDeck() {
  CudaCard hostDeck[52];
  for (int i = 0; i < 52; ++i) hostDeck[i] = getCudaCard(i);
  CHECK_CUDA_ERROR(cudaMemcpyToSymbol(REFERENCE_DECK, hostDeck, sizeof(CudaCard) * 52));

#if CUDA_DEBUG_REFERENCE_DECK
  printf("Reference Deck Initialization:\n");
  for (int i = 0; i < 52; ++i) {
    printf("PokerCard %2d: v0=%s, v1=%s, v2=%s, v3=%s\n",
           i,
           formatBinary16(hostDeck[i].v0).c_str(),
           formatBinary16(hostDeck[i].v1).c_str(),
           formatBinary16(hostDeck[i].v2).c_str(),
           formatBinary16(hostDeck[i].v3).c_str());
  }
#endif
}

__host__ __device__ inline int _popcount64(uint64_t x) {
  x = x - ((x >> 1) & 0x5555555555555555ULL);
  x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
  x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
  return (x * 0x0101010101010101ULL) >> 56;
}

#define popcount64(x) _mm_popcnt_u64(x)

__host__ PokerHandType getHandType_test(const std::vector<PokerCard>& hand) {
  CudaCard hostDeck[52];
  CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(hostDeck, REFERENCE_DECK, sizeof(CudaCard) * 52));

  uint64_t hv = hostDeck[static_cast<uint8_t>(hand[0])].v + hostDeck[static_cast<uint8_t>(hand[1])].v +
                hostDeck[static_cast<uint8_t>(hand[2])].v + hostDeck[static_cast<uint8_t>(hand[3])].v +
                hostDeck[static_cast<uint8_t>(hand[4])].v;

  //                                      3210987654321098765432109876543210987654321098765432109876543210
  uint64_t mask    = static_cast<uint64_t>(0b0000000000000001001001001000000000000000000000000000000000000000ULL);
  bool     isFlush = popcount64(hv & (hv >> 2) & mask);

  mask              = static_cast<uint64_t>(0b0000000000000000000000000000000000000000000000000001001001001001ULL);
  bool isStraight   = (5 == popcount64(hv & (mask << 24)));
  bool isRoyalFlush = isFlush && isStraight;
  isStraight |= (5 == popcount64(hv & (mask << 21)));
  isStraight |= (5 == popcount64(hv & (mask << 18)));
  isStraight |= (5 == popcount64(hv & (mask << 15)));
  isStraight |= (5 == popcount64(hv & (mask << 12)));
  isStraight |= (5 == popcount64(hv & (mask << 9)));
  isStraight |= (5 == popcount64(hv & (mask << 6)));
  isStraight |= (5 == popcount64(hv & (mask << 3)));
  isStraight |= (5 == popcount64(hv & mask));
  mask = static_cast<uint64_t>(0b0000000000000000000000000001000000000000000000000000001001001001ULL);
  isStraight |= (5 == popcount64(hv & mask));  // Ace low straight
  bool isStraightFlush = isStraight && isFlush;
  mask                 = static_cast<uint64_t>(0b0000000000000000000000000001001001001001001001001001001001001001ULL);
  bool isFourOfAKind   = (1 == popcount64(hv & (mask << 2)));
  bool isThreeOfAKind  = (1 == popcount64((hv + mask) & (mask << 2)));
  bool twoGroups       = (2 == popcount64(hv & (mask << 1)));
  bool oneGroup        = (1 == popcount64(hv & (mask << 1)));
  return isRoyalFlush      ? PokerHandType::RoyalFlush
         : isStraightFlush ? PokerHandType::StraightFlush
         : isFlush         ? PokerHandType::Flush
         : isStraight      ? PokerHandType::Straight
         : isFourOfAKind   ? PokerHandType::FourOfAKind
         : isThreeOfAKind  ? (twoGroups ? PokerHandType::FullHouse : PokerHandType::ThreeOfAKind)
         : twoGroups       ? PokerHandType::TwoPair
         : oneGroup        ? PokerHandType::OnePair
                           : PokerHandType::HighCard;
}

void initializeReferenceDeck64() {
  CudaCard hostDeck[52];
  for (int i = 0; i < 52; ++i) hostDeck[i] = getCudaCard64(i);
  CHECK_CUDA_ERROR(cudaMemcpyToSymbol(REFERENCE_DECK, hostDeck, sizeof(CudaCard) * 52));

#if CUDA_DEBUG_REFERENCE_DECK
  uint64_t mask1 = static_cast<uint64_t>(0b0000000000000001001001001000000000000000000000000000000000000000);
  printf("Mask %2d: %s\n", 1, formatBinary64(mask1).c_str());
  uint64_t mask2 = static_cast<uint64_t>(0b0000000000000000000000000000000000000000000000000001001001001001);
  printf("Mask %2d: %s\n", 2, formatBinary64(mask2).c_str());
  uint64_t mask3 = static_cast<uint64_t>(0b0000000000000000000000000001000000000000000000000000001001001001);
  printf("Mask %2d: %s\n", 3, formatBinary64(mask3).c_str());
  uint64_t mask4 = static_cast<uint64_t>(0b0000000000000000000000000001001001001001001001001001001001001001);
  printf("Mask %2d: %s\n", 4, formatBinary64(mask4).c_str());
  printf("Reference Deck Initialization:\n");
  for (int i = 0; i < 52; ++i) {
    printf("PokerCard %2d: %s\n", i, formatBinary64(hostDeck[i].v).c_str());
  }

  // Test popcount64
  uint64_t test = 0b000ULL;
  printf("popcount64(%s) = %d\n", formatBinary64(test).c_str(), popcount64(test));
  test = 0b001ULL;
  printf("popcount64(%s) = %d\n", formatBinary64(test).c_str(), popcount64(test));
  test = 0b010ULL;
  printf("popcount64(%s) = %d\n", formatBinary64(test).c_str(), popcount64(test));
  printf("popcount64(%s) = %d\n", formatBinary64(mask1).c_str(), popcount64(mask1));
  printf("popcount64(%s) = %d\n", formatBinary64(mask2).c_str(), popcount64(mask2));
  printf("popcount64(%s) = %d\n", formatBinary64(mask3).c_str(), popcount64(mask3));
  printf("popcount64(%s) = %d\n", formatBinary64(mask4).c_str(), popcount64(mask4));

  // Test hand types
  testHandType(royalFlush, PokerHandType::RoyalFlush, getHandType_test);
  testHandType(straightFlush, PokerHandType::StraightFlush, getHandType_test);
  testHandType(fourOfAKind, PokerHandType::FourOfAKind, getHandType_test);
  testHandType(fullHouse, PokerHandType::FullHouse, getHandType_test);
  testHandType(fullHouseA, PokerHandType::FullHouse, getHandType_test);
  testHandType(flush, PokerHandType::Flush, getHandType_test);
  testHandType(straight, PokerHandType::Straight, getHandType_test);
  testHandType(wheel, PokerHandType::Straight, getHandType_test);
  testHandType(broadway, PokerHandType::Straight, getHandType_test);
  testHandType(threeOfAKind, PokerHandType::ThreeOfAKind, getHandType_test);
  testHandType(threeAces, PokerHandType::ThreeOfAKind, getHandType_test);
  testHandType(twoPair, PokerHandType::TwoPair, getHandType_test);
  testHandType(onePair, PokerHandType::OnePair, getHandType_test);
  testHandType(twoAces, PokerHandType::OnePair, getHandType_test);
  testHandType(highCard, PokerHandType::HighCard, getHandType_test);

  std::cout << std::endl;

#endif
}