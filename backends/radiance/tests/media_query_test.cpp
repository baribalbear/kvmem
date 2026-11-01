#include "server/query_span.h"
#include <stdexcept>
#include <iostream>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while(false)
int main() {
    using namespace rad::server;
    const auto chars=[](const std::string& s){return std::vector<int32_t>(s.begin(),s.end());};
    const std::vector<std::vector<int32_t>> images{{1001,1002,1003},{2001,2002}};
    const auto encode=[&](const std::string& s){return encode_media_prefix(s,"<img>",images.size(),chars,
        [&](size_t i)->const std::vector<int32_t>&{return images[i];});};
    CHECK((encode("A<img>B")==std::vector<int32_t>{65,1001,1002,1003,66}));
    CHECK(encode("<img><img><img>").empty());
    const std::string history="U:A<img>B<img>\nA:OK\nU:",question="colors?",suffix="\nA:";
    const auto original=history+question+suffix,marked=history+"QUERY"+suffix;
    const auto tokens=encode(original);const auto span=prove_query_span(original,marked,"QUERY",question,tokens,encode);
    CHECK(span.first==int32_t(encode(history).size()));
    CHECK(span.second-span.first==int32_t(question.size()));
    CHECK(prove_history_end(original,history+question,tokens,encode)==span.second);
    auto changed=tokens;changed[4]++;
    CHECK(prove_query_span(original,marked,"QUERY",question,changed,encode).first==-1);
    CHECK(prove_history_end(original,history+question,changed,encode)==-1);
    const auto text="U:hello\nA:";const auto text_tokens=chars(text);
    CHECK(prove_query_span(text,"U:QUERY\nA:","QUERY","hello",text_tokens,chars).first==2);
    std::cout<<"media-expanded Query/history boundaries passed\n";
}
