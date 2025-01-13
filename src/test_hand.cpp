#include <iostream>
#include <vector>
#include "test.h"

PokerHandType getHandType_v1(const std::vector<PokerCard>& cards) {
  uint8_t suits[5] = {static_cast<uint8_t>(cards[0]) & 0x3u,
                      static_cast<uint8_t>(cards[1]) & 0x3u,
                      static_cast<uint8_t>(cards[2]) & 0x3u,
                      static_cast<uint8_t>(cards[3]) & 0x3u,
                      static_cast<uint8_t>(cards[4]) & 0x3u};
  uint8_t ranks[5] = {0};
  // Sort ranks in ascending order
  ranks[0] = static_cast<uint8_t>(cards[0]) >> 2;
  for (uint8_t i = 1; i < 5; ++i) {
    uint8_t newRank = static_cast<uint8_t>(cards[i]) >> 2;
    for (uint8_t j = i; j > 0; --j) {
      ranks[j] = newRank > ranks[j - 1] ? newRank : ranks[j - 1];
      newRank  = newRank > ranks[j - 1] ? ranks[j - 1] : newRank;
    }
    ranks[0] = newRank < ranks[0] ? newRank : ranks[0];
  }
  bool isFlush = ((suits[0] == suits[1]) && (suits[1] == suits[2]) && (suits[2] == suits[3]) && (suits[3] == suits[4]));
  bool isStraight = (((ranks[1] - ranks[0]) == 1) && ((ranks[2] - ranks[1]) == 1) && ((ranks[3] - ranks[2]) == 1) &&
                     (((ranks[4] - ranks[3]) == 1) || ((ranks[4] - ranks[3]) == 9)));
  if (isFlush)
    return isStraight ? (ranks[0] == 8) ? PokerHandType::RoyalFlush : PokerHandType::StraightFlush
                      : PokerHandType::Flush;
  if (isStraight) return PokerHandType::Straight;

  uint8_t numPair        = 0;
  bool    isThreeOfAKind = false;
  bool    isFourOfAKind  = false;

  for (int i = 0; i < 5; ++i) {
    int count = 1;
    for (int j = i + 1; j < 5; ++j) {
      if (ranks[i] == ranks[j]) {
        count++;
      }
    }
    if (count == 2) {
      numPair++;
    } else if (count == 3) {
      numPair--;
      isThreeOfAKind = true;
    } else if (count == 4) {
      isFourOfAKind = true;
    }
  }
  return isFourOfAKind    ? PokerHandType::FourOfAKind
         : isThreeOfAKind ? (numPair > 0) ? PokerHandType::FullHouse : PokerHandType::ThreeOfAKind
         : numPair == 2   ? PokerHandType::TwoPair
         : numPair == 1   ? PokerHandType::OnePair
                          : PokerHandType::HighCard;
}

int main() {
  testHandType(royalFlush, PokerHandType::RoyalFlush, getHandType_v1);
  testHandType(straightFlush, PokerHandType::StraightFlush, getHandType_v1);
  testHandType(fourOfAKind, PokerHandType::FourOfAKind, getHandType_v1);
  testHandType(fullHouse, PokerHandType::FullHouse, getHandType_v1);
  testHandType(fullHouseA, PokerHandType::FullHouse, getHandType_v1);
  testHandType(flush, PokerHandType::Flush, getHandType_v1);
  testHandType(straight, PokerHandType::Straight, getHandType_v1);
  testHandType(wheel, PokerHandType::Straight, getHandType_v1);
  testHandType(broadway, PokerHandType::Straight, getHandType_v1);
  testHandType(threeOfAKind, PokerHandType::ThreeOfAKind, getHandType_v1);
  testHandType(threeAces, PokerHandType::ThreeOfAKind, getHandType_v1);
  testHandType(twoPair, PokerHandType::TwoPair, getHandType_v1);
  testHandType(onePair, PokerHandType::OnePair, getHandType_v1);
  testHandType(twoAces, PokerHandType::OnePair, getHandType_v1);
  testHandType(highCard, PokerHandType::HighCard, getHandType_v1);

  std::cout << std::endl;

  return 0;
}
