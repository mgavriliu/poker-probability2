#pragma once

#include <nmmintrin.h>  // Add this include for _mm_popcnt_u64
#include <array>
#include <stdexcept>  // Add this include for std::runtime_error
#include <string>
#include <vector>

enum class PokerHandType : uint8_t {
  RoyalFlush = 0,
  StraightFlush,
  FourOfAKind,
  FullHouse,
  Flush,
  Straight,
  ThreeOfAKind,
  TwoPair,
  OnePair,
  HighCard,
  Count  // Used for iteration
};

enum class PokerCard : uint8_t {  // 0 to 51
  _2h = 0,
  _2d,
  _2c,
  _2s,
  _3h,
  _3d,
  _3c,
  _3s,
  _4h,
  _4d,
  _4c,
  _4s,
  _5h,
  _5d,
  _5c,
  _5s,
  _6h,
  _6d,
  _6c,
  _6s,
  _7h,
  _7d,
  _7c,
  _7s,
  _8h,
  _8d,
  _8c,
  _8s,
  _9h,
  _9d,
  _9c,
  _9s,
  Th,
  Td,
  Tc,
  Ts,
  Jh,
  Jd,
  Jc,
  Js,
  Qh,
  Qd,
  Qc,
  Qs,
  Kh,
  Kd,
  Kc,
  Ks,
  Ah,
  Ad,
  Ac,
  As
};

class Poker {
 public:
  static double getTheoreticalProbability(PokerHandType type) {
    return _handTypeProbabilities[static_cast<uint8_t>(type)];
  }
  static const char* toString(PokerHandType type) {
    switch (type) {
      case PokerHandType::RoyalFlush: return "Royal Flush";
      case PokerHandType::StraightFlush: return "Straight Flush";
      case PokerHandType::FourOfAKind: return "Four of a Kind";
      case PokerHandType::FullHouse: return "Full House";
      case PokerHandType::Flush: return "Flush";
      case PokerHandType::Straight: return "Straight";
      case PokerHandType::ThreeOfAKind: return "Three of a Kind";
      case PokerHandType::TwoPair: return "Two Pair";
      case PokerHandType::OnePair: return "One Pair";
      case PokerHandType::HighCard: return "High PokerCard";
      default: return "Unknown";
    }
  }
  static std::string toString(uint8_t card) {
    std::string rank = "";
    std::string suit = "";
    switch (card >> 2) {
      case 0: rank = "2"; break;
      case 1: rank = "3"; break;
      case 2: rank = "4"; break;
      case 3: rank = "5"; break;
      case 4: rank = "6"; break;
      case 5: rank = "7"; break;
      case 6: rank = "8"; break;
      case 7: rank = "9"; break;
      case 8: rank = "T"; break;
      case 9: rank = "J"; break;
      case 10: rank = "Q"; break;
      case 11: rank = "K"; break;
      case 12: rank = "A"; break;
    }
    switch (card & 0x3) {
      case 0: suit = "h"; break;
      case 1: suit = "d"; break;
      case 2: suit = "c"; break;
      case 3: suit = "s"; break;
    }
    return rank + suit;
  }

 private:
  static inline const std::array<double, static_cast<uint8_t>(PokerHandType::Count)> _handTypeProbabilities = {
      // Reference: https://en.wikipedia.org/wiki/Poker_probability
      100.0 * 1.0 / 649740.0,        // RoyalFlush
      100.0 * 9.0 / 649740.0,        // StraightFlush
      100.0 * 156.0 / 649740.0,      // FourOfAKind
      100.0 * 936.0 / 649740.0,      // FullHouse
      100.0 * 1277.0 / 649740.0,     // Flush
      100.0 * 2550.0 / 649740.0,     // Straight
      100.0 * 13728.0 / 649740.0,    // ThreeOfAKind
      100.0 * 30888.0 / 649740.0,    // TwoPair
      100.0 * 274560.0 / 649740.0,   // OnePair
      100.0 * 325635.0 / 649740.0};  // HighCard
};

class PokerHand {
 public:
  PokerHand() = default;
  PokerHand(const uint8_t handSize) : _handSize(handSize) {}
  PokerHand(uint8_t* cards, const uint8_t handSize = 5) : _handSize(handSize) {
    for (uint8_t i = 0; i < _handSize; ++i) {
      addCard(cards[i]);
    }
    this->ComputeHandRank();
  }
  PokerHand(std::vector<PokerCard> cards, const uint8_t handSize = 5) : _handSize(handSize) {
    if (cards.size() != _handSize) {
      throw std::runtime_error("PokerHand must contain exactly " + std::to_string(_handSize) + " cards");
    }
    for (const PokerCard& card : cards) {
      addCard(static_cast<uint8_t>(card));
    }
    this->ComputeHandRank();
  }

  void addCard(const uint8_t& card) {
    _cards.push_back(card);
    _asString += Poker::toString(card);
    _cardsInHand++;
    if (_cardsInHand == _handSize) {
      this->ComputeHandRank();
    }
  }

  virtual inline PokerHandType ComputeHandRank() {
    uint8_t suits[5] = {static_cast<uint8_t>(_cards[0] & 0x3),
                        static_cast<uint8_t>(_cards[1] & 0x3),
                        static_cast<uint8_t>(_cards[2] & 0x3),
                        static_cast<uint8_t>(_cards[3] & 0x3),
                        static_cast<uint8_t>(_cards[4] & 0x3)};
    uint8_t ranks[5] = {0};
    // Sort ranks in ascending order
    ranks[0] = static_cast<uint8_t>(_cards[0] >> 2);
#pragma unroll
    for (uint8_t i = 1; i < 5; ++i) {
      uint8_t newRank = static_cast<uint8_t>(_cards[i] >> 2);
#pragma unroll
      for (uint8_t j = i; j > 0; --j) {
        ranks[j] = newRank > ranks[j - 1] ? newRank : ranks[j - 1];
        newRank  = newRank > ranks[j - 1] ? ranks[j - 1] : newRank;
      }
      ranks[0] = newRank < ranks[0] ? newRank : ranks[0];
    }
    bool isFlush =
        ((suits[0] == suits[1]) && (suits[1] == suits[2]) && (suits[2] == suits[3]) && (suits[3] == suits[4]));
    bool isStraight = (((ranks[1] - ranks[0]) == 1) && ((ranks[2] - ranks[1]) == 1) && ((ranks[3] - ranks[2]) == 1) &&
                       (((ranks[4] - ranks[3]) == 1) || ((ranks[4] - ranks[3]) == 9)));

    uint8_t numPair        = 0;
    bool    isThreeOfAKind = false;
    bool    isFourOfAKind  = false;

#pragma unroll
    for (int i = 0; i < 5; ++i) {
      int count = 1;
#pragma unroll
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
    bool isStraightFlush = isFlush && isStraight;
    bool isRoyalFlush    = isStraightFlush && (ranks[4] == 8);
    this->_handType      = isRoyalFlush      ? PokerHandType::RoyalFlush
                           : isStraightFlush ? PokerHandType::StraightFlush
                           : isFlush         ? PokerHandType::Flush
                           : isStraight      ? PokerHandType::Straight
                           : isFourOfAKind   ? PokerHandType::FourOfAKind
                           : isThreeOfAKind  ? (numPair > 0) ? PokerHandType::FullHouse : PokerHandType::ThreeOfAKind
                           : numPair == 2    ? PokerHandType::TwoPair
                           : numPair == 1    ? PokerHandType::OnePair
                                             : PokerHandType::HighCard;
    return this->_handType;
  }
  PokerHandType getHandType() const { return _handType; }
  std::string   toString() const { return _asString; }

 private:
  uint8_t              _handSize    = 5;
  uint8_t              _cardsInHand = 0;
  std::vector<uint8_t> _cards;
  PokerHandType        _handType = PokerHandType::HighCard;
  std::string          _asString = "";
};

struct PokerHand64 {
  union {
    struct {
      uint16_t v3;
      uint16_t v2;
      uint16_t v1;
      uint16_t v0;
    };
    uint64_t v = 0x0ull;
  };
  uint8_t _handSize = 5;

  inline PokerHand64() = default;

  PokerHand64(uint8_t* cards, const uint8_t handSize = 5) : _handSize(handSize) {
    for (uint8_t i = 0; i < _handSize; ++i) {
      this->v += fromPokerHand(cards[i]);
    }
  }
  PokerHand64(std::vector<PokerCard> cards, const uint8_t handSize = 5) : _handSize(handSize) {
    if (cards.size() != _handSize) {
      throw std::runtime_error("PokerHand must contain exactly " + std::to_string(_handSize) + " cards");
    }
    for (uint8_t i = 0; i < _handSize; ++i) {
      this->v += fromPokerHand(static_cast<uint8_t>(cards[i]));
    }
  }

  inline uint64_t fromPokerHand(uint8_t card) {
    // Version with single 64 bit storage.
    // Each rank and suit get 3 bits.
    // Ranks start at 0 and end at 38
    // Suits start at 39 and end at 50
    uint8_t  r = card >> 2;
    uint8_t  s = card & 0x3;
    uint64_t c;
    c = static_cast<uint64_t>(static_cast<uint64_t>(0x1) << (3 * s));   // First pack in the suit
    c <<= 39;                                                           // Shift to the suit
    c |= static_cast<uint64_t>(static_cast<uint64_t>(0x1) << (3 * r));  // Add the rank

    return c;
  }

  inline PokerHandType ComputeHandRank() {
    //                                          3210987654321098765432109876543210987654321098765432109876543210
    uint64_t mask     = static_cast<uint64_t>(0b0000000000000001001001001000000000000000000000000000000000000000ULL);
    bool     isFlush  = _mm_popcnt_u64(v & (v >> 2) & mask);
    mask              = static_cast<uint64_t>(0b0000000000000000000000000000000000000000000000000001001001001001ULL);
    bool isStraight   = (5 == _mm_popcnt_u64(v & (mask << 24)));
    bool isRoyalFlush = isFlush && isStraight;
    isStraight |= (5 == _mm_popcnt_u64(v & (mask << 21)));
    isStraight |= (5 == _mm_popcnt_u64(v & (mask << 18)));
    isStraight |= (5 == _mm_popcnt_u64(v & (mask << 15)));
    isStraight |= (5 == _mm_popcnt_u64(v & (mask << 12)));
    isStraight |= (5 == _mm_popcnt_u64(v & (mask << 9)));
    isStraight |= (5 == _mm_popcnt_u64(v & (mask << 6)));
    isStraight |= (5 == _mm_popcnt_u64(v & (mask << 3)));
    isStraight |= (5 == _mm_popcnt_u64(v & mask));
    mask = static_cast<uint64_t>(0b0000000000000000000000000001000000000000000000000000001001001001ULL);
    isStraight |= (5 == _mm_popcnt_u64(v & mask));  // Ace low straight
    bool isStraightFlush = isStraight && isFlush;
    mask                 = static_cast<uint64_t>(0b0000000000000000000000000001001001001001001001001001001001001001ULL);
    bool isFourOfAKind   = (1 == _mm_popcnt_u64(v & (mask << 2)));
    bool isThreeOfAKind  = (1 == _mm_popcnt_u64((v + mask) & (mask << 2)));
    bool twoGroups       = (2 == _mm_popcnt_u64(v & (mask << 1)));
    bool oneGroup        = (1 == _mm_popcnt_u64(v & (mask << 1)));
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
};
